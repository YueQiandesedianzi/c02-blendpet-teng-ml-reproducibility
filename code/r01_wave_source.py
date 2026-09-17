from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


ANCHORS = np.asarray([0.0, 25.0, 50.0, 75.0, 100.0], dtype=np.float32)
PHYSICAL_NAMES = (
    "log1p_vpp",
    "signed_dominant_peak_over_vpp",
    "peak_balance",
    "rms_over_vpp",
    "signed_area_balance",
    "half_peak_width_fraction",
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))


def robust_scale(values: np.ndarray, minimum: float = 1e-6) -> float:
    values = np.asarray(values, dtype=float).reshape(-1)
    center = float(np.median(values))
    scale = float(1.4826 * np.median(np.abs(values - center)))
    return max(scale, minimum)


def robust_columns(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    center = np.median(values, axis=0).astype(np.float32)
    scale = (1.4826 * np.median(np.abs(values - center), axis=0)).astype(np.float32)
    scale = np.where(scale < 1e-6, 1.0, scale).astype(np.float32)
    return center, scale


def edge_linear_correct(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    count = max(4, int(round(0.10 * len(values))))
    left_x = 0.5 * (count - 1)
    right_x = len(values) - 0.5 * (count + 1)
    left = float(np.median(values[:count]))
    right = float(np.median(values[-count:]))
    slope = (right - left) / max(right_x - left_x, 1.0)
    baseline = left + slope * (np.arange(len(values), dtype=np.float32) - left_x)
    return (values - baseline).astype(np.float32)


def align_dominant_peak(values: np.ndarray, peak_index: int = 128) -> np.ndarray:
    corrected = edge_linear_correct(values)
    source = int(np.argmax(np.abs(corrected)))
    return np.roll(corrected, peak_index - source).astype(np.float32)


def build_aligned_waves(raw: np.ndarray) -> np.ndarray:
    return np.asarray([align_dominant_peak(row) for row in np.asarray(raw)], dtype=np.float32)


def half_peak_width_fraction(wave: np.ndarray) -> float:
    values = np.asarray(wave, dtype=float)
    peak = int(np.argmax(np.abs(values)))
    threshold = 0.5 * abs(float(values[peak]))
    if threshold <= 1e-12:
        return 0.0
    left = peak
    while left > 0 and abs(values[left - 1]) >= threshold:
        left -= 1
    right = peak
    while right + 1 < len(values) and abs(values[right + 1]) >= threshold:
        right += 1
    return float(right - left + 1) / float(len(values))


def physical_features_numpy(waves: np.ndarray) -> np.ndarray:
    waves = np.asarray(waves, dtype=np.float32)
    maximum = np.max(waves, axis=1)
    minimum = np.min(waves, axis=1)
    positive_peak = np.maximum(maximum, 0.0)
    negative_peak = np.maximum(-minimum, 0.0)
    vpp = np.maximum(maximum - minimum, 1e-9)
    dominant = np.where(np.abs(maximum) >= np.abs(minimum), maximum, minimum)
    rms = np.sqrt(np.mean(waves**2, axis=1))
    positive_area = np.sum(np.maximum(waves, 0.0), axis=1)
    negative_area = np.sum(np.maximum(-waves, 0.0), axis=1)
    area_total = np.maximum(positive_area + negative_area, 1e-9)
    widths = np.asarray([half_peak_width_fraction(row) for row in waves], dtype=np.float32)
    return np.column_stack(
        [
            np.log1p(vpp),
            dominant / vpp,
            (positive_peak - negative_peak) / vpp,
            rms / vpp,
            (positive_area - negative_area) / area_total,
            widths,
        ]
    ).astype(np.float32)


def channel_tensor(wave: torch.Tensor, amplitude_scale: float) -> torch.Tensor:
    vpp = torch.clamp(torch.amax(wave, dim=1) - torch.amin(wave, dim=1), min=1e-9)
    amplitude = wave / float(amplitude_scale)
    shape = torch.abs(wave) / vpp[:, None]
    return torch.stack([amplitude, shape], dim=1)


def physical_features_torch(wave: torch.Tensor) -> torch.Tensor:
    maximum = torch.amax(wave, dim=1)
    minimum = torch.amin(wave, dim=1)
    positive_peak = torch.clamp(maximum, min=0.0)
    negative_peak = torch.clamp(-minimum, min=0.0)
    vpp = torch.clamp(maximum - minimum, min=1e-9)
    dominant = torch.where(torch.abs(maximum) >= torch.abs(minimum), maximum, minimum)
    rms = torch.sqrt(torch.mean(wave**2, dim=1))
    positive_area = torch.sum(torch.clamp(wave, min=0.0), dim=1)
    negative_area = torch.sum(torch.clamp(-wave, min=0.0), dim=1)
    area_total = torch.clamp(positive_area + negative_area, min=1e-9)
    absolute = torch.abs(wave)
    peak = torch.argmax(absolute, dim=1)
    peak_abs = torch.gather(absolute, 1, peak[:, None]).reshape(-1)
    above = absolute >= (0.5 * peak_abs[:, None])
    positions = torch.arange(wave.shape[1], device=wave.device)[None, :]
    left_invalid = torch.where(
        (positions < peak[:, None]) & (~above), positions, positions.new_full(positions.shape, -1)
    ).amax(dim=1)
    right_invalid = torch.where(
        (positions > peak[:, None]) & (~above), positions, positions.new_full(positions.shape, wave.shape[1])
    ).amin(dim=1)
    width = (right_invalid - left_invalid - 1).to(wave.dtype) / float(wave.shape[1])
    width = torch.where(peak_abs > 1e-12, width, torch.zeros_like(width))
    return torch.stack(
        [
            torch.log1p(vpp),
            dominant / vpp,
            (positive_peak - negative_peak) / vpp,
            rms / vpp,
            (positive_area - negative_area) / area_total,
            width,
        ],
        dim=1,
    )


def augment_wave(wave: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    result = wave.clone()
    batch, points = result.shape
    shifts = torch.randint(-4, 5, (batch,), generator=generator, device=result.device)
    result = torch.stack([torch.roll(result[i], int(shifts[i])) for i in range(batch)])
    time_scale = 0.98 + 0.04 * torch.rand(batch, generator=generator, device=result.device)
    base = torch.linspace(-1.0, 1.0, points, device=result.device)
    grid_x = torch.clamp(base[None, :] / time_scale[:, None], -1.0, 1.0)
    grid = torch.stack([grid_x, torch.zeros_like(grid_x)], dim=-1).reshape(batch, 1, points, 2)
    result = F.grid_sample(
        result.reshape(batch, 1, 1, points),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    ).reshape(batch, points)
    amplitude = 0.98 + 0.04 * torch.rand(batch, generator=generator, device=result.device)
    result = result * amplitude[:, None]
    vpp = torch.clamp(torch.amax(result, dim=1) - torch.amin(result, dim=1), min=1e-9)
    noise_fraction = 0.005 * torch.rand(batch, generator=generator, device=result.device)
    noise = torch.randn(result.shape, generator=generator, device=result.device, dtype=result.dtype)
    return result + noise * (noise_fraction * vpp)[:, None]


def balanced_source_composition_indices(
    indices: np.ndarray,
    labels: np.ndarray,
    sources: np.ndarray,
    seed: int,
) -> np.ndarray:
    indices = np.asarray(indices, dtype=int)
    labels = np.asarray(labels, dtype=float)
    sources = np.asarray(sources, dtype=str)
    rng = np.random.default_rng(seed)
    composition_sets: list[np.ndarray] = []
    for composition in sorted(np.unique(labels[indices]).tolist()):
        current = indices[np.isclose(labels[indices], composition)]
        source_names = sorted(np.unique(sources[current]).tolist())
        per_source = max(int(np.sum(sources[current] == source)) for source in source_names)
        source_balanced = np.concatenate(
            [
                rng.choice(
                    current[sources[current] == source],
                    size=per_source,
                    replace=np.sum(sources[current] == source) < per_source,
                )
                for source in source_names
            ]
        ).astype(int)
        composition_sets.append(source_balanced)
    per_composition = max(len(values) for values in composition_sets)
    balanced = np.concatenate(
        [rng.choice(values, size=per_composition, replace=len(values) < per_composition) for values in composition_sets]
    ).astype(int)
    return rng.permutation(balanced)


class PhysicsOrdinalCNN(nn.Module):
    def __init__(self, wave_enabled: bool = True, physics_enabled: bool = True) -> None:
        super().__init__()
        if not wave_enabled and not physics_enabled:
            raise ValueError("Enable the waveform path or the physics path.")
        self.wave_enabled = bool(wave_enabled)
        self.physics_enabled = bool(physics_enabled)
        self.conv1 = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=9, padding=4),
            nn.GroupNorm(4, 16),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.wave_projection = nn.Linear(32, 8)
        self.wave_head = nn.Linear(8, 1)
        self.physics_beta = nn.Parameter(torch.zeros(len(PHYSICAL_NAMES)))
        self.evidence_bias = nn.Parameter(torch.tensor(0.0))
        self.threshold_base = nn.Parameter(torch.tensor(-1.5))
        inverse_softplus_one = math.log(math.expm1(1.0))
        self.threshold_delta_raw = nn.Parameter(torch.full((3,), inverse_softplus_one))

    def thresholds(self) -> torch.Tensor:
        increments = F.softplus(self.threshold_delta_raw) + 1e-4
        return torch.cat([self.threshold_base.reshape(1), self.threshold_base + torch.cumsum(increments, dim=0)])

    def waveform_evidence(self, channels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values = self.conv1(channels)
        values = self.conv2(values)
        pooled = F.adaptive_avg_pool1d(values, 1).flatten(1)
        latent = self.wave_projection(pooled)
        evidence = self.wave_head(latent).reshape(-1)
        if not self.wave_enabled:
            evidence = torch.zeros_like(evidence)
        return evidence, latent

    def components(
        self, channels: torch.Tensor, physical: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        wave_evidence, latent = self.waveform_evidence(channels)
        physics_evidence = torch.sum(physical * self.physics_beta[None, :], dim=1)
        if not self.physics_enabled:
            physics_evidence = torch.zeros_like(physics_evidence)
        evidence = wave_evidence + physics_evidence + self.evidence_bias
        thresholds = self.thresholds()
        ordinal_logits = evidence[:, None] - thresholds[None, :]
        exceedance = torch.sigmoid(ordinal_logits)
        probability = torch.stack(
            [
                1.0 - exceedance[:, 0],
                exceedance[:, 0] - exceedance[:, 1],
                exceedance[:, 1] - exceedance[:, 2],
                exceedance[:, 2] - exceedance[:, 3],
                exceedance[:, 3],
            ],
            dim=1,
        )
        prediction = 25.0 * torch.sum(exceedance, dim=1)
        return {
            "prediction": prediction,
            "probability": probability,
            "exceedance": exceedance,
            "ordinal_logits": ordinal_logits,
            "evidence": evidence,
            "wave_evidence": wave_evidence,
            "physics_evidence": physics_evidence,
            "latent": latent,
            "thresholds": thresholds,
        }

    def forward(self, channels: torch.Tensor, physical: torch.Tensor) -> torch.Tensor:
        return self.components(channels, physical)["prediction"]


@dataclass
class TrainedOrdinal:
    model: PhysicsOrdinalCNN
    amplitude_scale: float
    physical_center: np.ndarray
    physical_scale: np.ndarray
    best_epoch: int
    validation_mae: float
    history: list[dict[str, float]]
    seed: int
    family: str
    variant: str


def composition_equal_mae(target: np.ndarray, prediction: np.ndarray) -> float:
    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    return float(np.mean([np.mean(np.abs(prediction[np.isclose(target, value)] - value)) for value in np.unique(target)]))


def ordinal_targets(target: torch.Tensor) -> torch.Tensor:
    boundaries = target.new_tensor([25.0, 50.0, 75.0, 100.0])
    return (target[:, None] >= boundaries[None, :]).to(target.dtype)


@torch.no_grad()
def predict_model(
    trained: TrainedOrdinal,
    waves: np.ndarray,
    physical: np.ndarray,
    indices: np.ndarray,
    batch_size: int = 256,
) -> dict[str, np.ndarray]:
    trained.model.eval()
    device = next(trained.model.parameters()).device
    indices = np.asarray(indices, dtype=int)
    pieces: dict[str, list[np.ndarray]] = {
        "prediction": [], "probability": [], "exceedance": [], "evidence": [],
        "wave_evidence": [], "physics_evidence": [], "latent": [],
    }
    for start in range(0, len(indices), batch_size):
        current = indices[start : start + batch_size]
        wave = torch.tensor(waves[current], dtype=torch.float32, device=device)
        features = torch.tensor(
            (physical[current] - trained.physical_center) / trained.physical_scale,
            dtype=torch.float32,
            device=device,
        )
        output = trained.model.components(channel_tensor(wave, trained.amplitude_scale), features)
        for key in pieces:
            pieces[key].append(output[key].detach().cpu().numpy())
    result = {key: np.concatenate(value, axis=0).astype(float) for key, value in pieces.items()}
    result["thresholds"] = trained.model.thresholds().detach().cpu().numpy().astype(float)
    result["physics_beta"] = trained.model.physics_beta.detach().cpu().numpy().astype(float)
    return result


def fit_model(
    *,
    waves: np.ndarray,
    physical: np.ndarray,
    labels: np.ndarray,
    sources: np.ndarray,
    train_idx: np.ndarray,
    validation_idx: np.ndarray,
    seed: int,
    family: str,
    variant: str,
    max_epochs: int,
    patience: int,
    batch_size: int = 64,
) -> TrainedOrdinal:
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    waves = np.asarray(waves, dtype=np.float32)
    physical = np.asarray(physical, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    sources = np.asarray(sources, dtype=str)
    train_idx = np.asarray(train_idx, dtype=int)
    validation_idx = np.asarray(validation_idx, dtype=int)
    amplitude_scale = robust_scale(waves[train_idx].reshape(-1))
    physical_center, physical_scale = robust_columns(physical[train_idx])
    wave_enabled = variant != "physics-only-ordinal-head"
    physics_enabled = variant != "wave-only-ordinal-cnn"
    model = PhysicsOrdinalCNN(wave_enabled=wave_enabled, physics_enabled=physics_enabled).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    center_tensor = torch.tensor(physical_center, dtype=torch.float32, device=device)
    scale_tensor = torch.tensor(physical_scale, dtype=torch.float32, device=device)
    best_state: dict[str, torch.Tensor] | None = None
    best_value = math.inf
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, max_epochs + 1):
        epoch_idx = balanced_source_composition_indices(train_idx, labels, sources, seed + epoch * 1009)
        loader = DataLoader(
            TensorDataset(
                torch.tensor(waves[epoch_idx], dtype=torch.float32),
                torch.tensor(labels[epoch_idx], dtype=torch.float32),
            ),
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed + epoch),
        )
        model.train()
        losses: list[float] = []
        augmentation_generator = torch.Generator(device=device).manual_seed(seed + epoch * 3571)
        for wave, target in loader:
            wave = wave.to(device)
            target = target.to(device)
            augmented = augment_wave(wave, augmentation_generator)
            augmented_physical = physical_features_torch(augmented)
            output = model.components(
                channel_tensor(augmented, amplitude_scale),
                (augmented_physical - center_tensor) / scale_tensor,
            )
            huber = F.huber_loss(output["prediction"], target, delta=5.0)
            ordinal_bce = F.binary_cross_entropy_with_logits(
                output["ordinal_logits"], ordinal_targets(target)
            )
            loss = 0.7 * huber + 0.3 * ordinal_bce
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        trained = TrainedOrdinal(
            model=model,
            amplitude_scale=amplitude_scale,
            physical_center=physical_center,
            physical_scale=physical_scale,
            best_epoch=epoch,
            validation_mae=float("nan"),
            history=[],
            seed=seed,
            family=family,
            variant=variant,
        )
        validation_prediction = predict_model(trained, waves, physical, validation_idx, batch_size)["prediction"]
        metric = composition_equal_mae(labels[validation_idx], validation_prediction)
        history.append({"epoch": float(epoch), "training_loss": float(np.mean(losses)), "validation_mae": metric})
        if metric < best_value - 1e-7:
            best_value = metric
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("Training did not produce a valid model state.")
    model.load_state_dict(best_state)
    return TrainedOrdinal(
        model=model,
        amplitude_scale=amplitude_scale,
        physical_center=physical_center,
        physical_scale=physical_scale,
        best_epoch=best_epoch,
        validation_mae=best_value,
        history=history,
        seed=seed,
        family=family,
        variant=variant,
    )


def ensemble_predict(
    models: list[TrainedOrdinal],
    waves: np.ndarray,
    physical: np.ndarray,
    indices: np.ndarray,
) -> dict[str, np.ndarray]:
    outputs = [predict_model(model, waves, physical, indices) for model in models]
    array_keys = ("prediction", "probability", "exceedance", "evidence", "wave_evidence", "physics_evidence", "latent")
    return {key: np.mean(np.stack([output[key] for output in outputs], axis=0), axis=0) for key in array_keys}


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def checkpoint_payload(trained: TrainedOrdinal) -> dict[str, object]:
    return {
        "format_version": "v19",
        "model_state_dict": {key: value.detach().cpu() for key, value in trained.model.state_dict().items()},
        "family": trained.family,
        "variant": trained.variant,
        "seed": trained.seed,
        "amplitude_scale": float(trained.amplitude_scale),
        "physical_center": trained.physical_center.astype(float).tolist(),
        "physical_scale": trained.physical_scale.astype(float).tolist(),
        "physical_names": list(PHYSICAL_NAMES),
        "input_shape": [2, 256],
        "anchors": ANCHORS.astype(float).tolist(),
        "best_epoch": int(trained.best_epoch),
        "validation_mae": float(trained.validation_mae),
        "parameter_count": parameter_count(trained.model),
    }


def load_checkpoint(path: str | bytes | "os.PathLike[str]") -> TrainedOrdinal:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    variant = str(payload["variant"])
    model = PhysicsOrdinalCNN(
        wave_enabled=variant != "physics-only-ordinal-head",
        physics_enabled=variant != "wave-only-ordinal-cnn",
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return TrainedOrdinal(
        model=model,
        amplitude_scale=float(payload["amplitude_scale"]),
        physical_center=np.asarray(payload["physical_center"], dtype=np.float32),
        physical_scale=np.asarray(payload["physical_scale"], dtype=np.float32),
        best_epoch=int(payload["best_epoch"]),
        validation_mae=float(payload["validation_mae"]),
        history=[],
        seed=int(payload["seed"]),
        family=str(payload["family"]),
        variant=variant,
    )
