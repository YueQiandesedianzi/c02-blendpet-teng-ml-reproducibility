# CNN–P–V reproducibility

This repository provides the frozen analysis for composition prediction in PGD–PGS and PGT–PGD blends.
Release [rev23-v03](https://github.com/YueQiandesedianzi/c02-blendpet-teng-ml-reproducibility/releases/tag/rev23-v03) accompanies the revised machine-learning supplement.
The two systems were trained separately. This release changes presentation and access. It does not retrain models.

## Download

Download the three named assets from the release page:

| Asset | Contents |
| --- | --- |
| c02-blendpet-teng-supplementary-data-1-v03-2026-09-17.xlsx | Exact filterable tables and a file index. |
| c02-blendpet-teng-supplementary-data-2-v03-2026-09-17.zip | Cycle errors, saved bootstrap results, QC ledgers, and additional figures. |
| c02-blendpet-teng-supplementary-code-1-v03-2026-09-17.zip | Complete numerical code, frozen inputs, models, logs, and environment records. |

Download `manifest-sha256.csv` from the same release. It lists each asset digest.
On Windows, run `certutil -hashfile FILE SHA256` for each downloaded asset. Compare the result with the manifest.
On Linux or macOS, use `sha256sum FILE` or `shasum -a 256 FILE`.
Each ZIP also contains an internal `manifest-sha256.csv` for its payload files.
Extract Code 1 into a new directory. Run the following commands from that directory.
The Git checkout alone does not contain the large data and model files.

## Environment

The recorded run used Python 3.14.3, PyTorch 2.12.0+cpu, NumPy 2.4.6, pandas 3.0.3, SciPy 1.17.1, and scikit-learn 1.8.0.
Inspect `requirements.txt` and `provenance/environment.json`. The package does not include an interpreter or third-party dependencies.
Create a separate environment and install those dependencies. Exact historical versions may require the corresponding package index.

```text
python -m venv env
env/Scripts/python -m pip install -r requirements.txt
```

On Linux or macOS, replace `env/Scripts/python` with `env/bin/python`.
In the commands below, `python` refers to this environment.

## Check without retraining

```text
python code/check_release.py
```

This command verifies all payload hashes, replays the saved final controls, and checks R01 against frozen predictions.
It does not train models, calculate errors for model selection, or recompute bootstrap intervals.
It writes `qa/release-check-current.json`. The tolerance is 0.001 composition percentage points (pp).

## Predict

```text
python code/predict.py --input data/inference-pgd-pgs.npz --family PGD-PGS --output outputs/pgd-pgs.csv
python code/predict.py --input data/inference-pgt-pgd.npz --family PGT-PGD --output outputs/pgt-pgd.csv
```

The prediction entry point loads no target labels. Inputs contain `waves`, `base`, `base_names`, and `analysis_cycle_id`.
Waveforms have 256 points. Metadata definitions are in `results/feature-dictionary.csv` and `docs/methods.md`.
Exact R01 replay additionally requires the supplied historical metadata. A new cycle alone cannot reconstruct that metadata.

## Other entry points

Use a working copy for commands that regenerate files. Keep the downloaded archive as the frozen reference.

```text
python code/reproduce.py replay-r01
python code/reproduce.py evaluate
python code/reproduce.py retrain --out ../cnn-p-v-retrain
```

`replay-r01` writes replay records. `evaluate` calculates errors from saved predictions.
`retrain` requires a directory that does not exist. It saves the protocol before training and target-free predictions before evaluation.
It trains all fixed controls, then recalculates metrics and bootstrap summaries.
The inherited `verify` command and `code/audit_and_summarize.py` also regenerate bootstrap summaries. Use `check_release.py` for a check without that calculation.
The unchanged `code/compact-figures.py` draws the older supplemental layout. Its S2–S4 labels refer to the previous SI.
The current SI uses the final-inference figure in Data 2. `docs/file-index.md` maps all figure versions.

## Evidence and limits

The pool contains 2,227 analysis records and 2,000 unique physical cycles. Both systems share 227 pure-PGD cycles.
The 10%, 45%, and 60% compositions are `REUSED_CONFIRMATION`. Earlier development examined them.
Historical anchors are audits. OOF CNN predictions alone do not establish independent evaluation of the later calibration.
New controls use three temporal blocks per anchor file, an adjacent-cycle embargo, and training-only learned fits.
Historical QC still uses file context. The exact acquisition-to-QC version chain is incomplete.
Within-file evaluation does not establish generalization to new films, batches, or devices.
Historical R04 lacks its fitted head. Available predictions do not repair that gap.
The earlier v02 study used a different protocol. Its selected configurations passed only 1/6 confirmation checks.
All eight controls and negative results remain available. Fusion did not improve every metric or system.

## Citation and license

Use `CITATION.cff` and cite this release version. No DOI is assigned in this repository.
Code uses MIT. Research data, results, figures, model states, and documentation use CC BY 4.0.
Third-party materials retain their original terms. See `LICENSE`, `LICENSES/CC-BY-4.0.txt`, and `THIRD_PARTY_NOTICES.md`.
