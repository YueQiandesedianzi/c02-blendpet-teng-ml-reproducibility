# Method and result reference

This text accompanies the SI. Native equations remain editable in the submitted SI.


## Supplementary Note S1 Data and preprocessing

The source manifest contains 30 files. Historical QC retained 2,026 of 2,215 candidate physical cycles. A later edge-mismatch filter excluded 26 cycles. The final pool contains 2,000 unique cycles and 2,227 analysis records: 1,075 for PGD–PGS and 1,152 for PGT–PGD. Both systems share 227 pure-PGD cycles. Table S5 summarizes counts and exclusion reasons. No cycle was excluded because of new prediction errors.

### Historical acquisition and QC provenance

Verified numerical reproduction starts from the frozen 256-point cycles and their metadata. Historical QC used source-file context. The available raw-file and script versions do not reproduce the complete frozen candidate ledger. Supplementary Code 1 preserves the archived upstream parameters in notes/archived-upstream-parameters.md; these settings are not asserted to match every frozen file. Supplementary Note S6 states the evaluation limits.

### Verified order after the frozen resampling

A frozen cycle r has N=256 voltage points. The historical edge-mismatch ratio is the absolute difference between the first and last 26-point medians divided by max(VPP,10⁻¹² V). The final pool retains ratios at most 0.25. Local correction estimates a line through these two medians at point indices 12.5 and 242.5. It subtracts the line before aligning the dominant peak. P16 amplitude, area, and shape statistics use the corrected unshifted cycle. P6 discrete area balance and contiguous width use the aligned cycle. No additional 256-point resampling follows this alignment in the supplied R01 implementation.

`xj = rj − [mL + (mR − mL)(j − 12.5)/230],  j=0,…,255`

`w = roll(x, 128 − argmaxj |xj|)`

The first maximum is used on an exact tie. Alignment is a cyclic shift and preserves the point count. CNN channel 1 is w/a, where a=max(1.4826 median|w−median(w)|,10⁻⁶ V), computed over training-wave values. Channel 2 is |w|/max(VPP,10⁻⁹ V). R01 uses its saved seed-specific amplitude scales. New controls fit a only on the current training portion. The preprocessing example is in Supplementary Data 2, figures/preprocessing.svg (release rev23-v03).

## Supplementary Note S2 Physical descriptor definitions

Table S1 defines the complete P6 and P16 sets. Here x is the corrected unshifted cycle, w is its aligned form, p=max(x), n=min(x), V=p−n, T is the recorded cycle duration, and x⁺=max(x,0), x⁻=max(−x,0). All input values must be finite. The dictionaries and implementations are in code/preprocessing.py and reference/raw-processing/data_pipeline/feature_engineering.py. Categories describe physical meaning: A, amplitude and peak shape; B, area; C, time and width; D, waveform statistics; E, reference and signal quality.

P6 is used during original CNN training in both systems and in final PGD–PGS calibration. P16 is used in final PGT–PGD calibration. The training P6 branch uses per-column medians and 1.4826 MAD; scales below 10⁻⁶ become 1. Final scaler settings are listed in Table S3. The new controls use StandardScaler fitted on the current training part: (X−mean)/population SD, with scale 1 for a constant column. There is no feature selection or search.

### Table S1a P6 dictionary

| Field / category | Formula | Unit and calculation | Numerical rule |
| --- | --- | --- | --- |
| log1p_vpp [A] | ln[1 + max(V, 10⁻⁹ V)/(1 V)] | 1; Corrected, before alignment | V is stored in volts. |
| signed_dominant_peak_over_vpp [A] | d / max(V, 10⁻⁹ V) | 1; Corrected, before alignment | d = p if \|p\| ≥ \|n\|; otherwise n. |
| peak_balance [A] | (max(p,0) − max(−n,0)) / max(V,10⁻⁹ V) | 1; Corrected, before alignment | p=max(x); n=min(x). |
| rms_over_vpp [A] | sqrt(mean(x²)) / max(V,10⁻⁹ V) | 1; Corrected, before alignment | Amplitude ratio. |
| signed_area_balance [B] | (Σx⁺ − Σx⁻) / max(Σ\|x\|,10⁻⁹ V) | 1; Aligned 256-point cycle | Point sums; not a voltage–time integral. |
| half_peak_width_fraction [C] | (R − L + 1) / 256 | 1; Aligned 256-point cycle | Contiguous \|x\| ≥ \|xpeak\|/2 around peak; no wrap. Zero if peak ≤10⁻¹² V. |

The P6 area balance is a ratio of discrete sums. It is not a trapezoidal voltage–time integral. Its width counts only the contiguous half-height region containing the dominant peak, divided by 256. It is not the sum of all half-height regions in a cycle.

## Table S1b P16 dictionary

| Field / category | Formula | Unit | Calculation and guard |
| --- | --- | --- | --- |
| vpp_v [A] | max(x) − min(x) | V | Corrected, before alignment. VPP; no transform. |
| positive_peak_v [A] | max(x) | V | Corrected, before alignment. Signed maximum, not clamped at zero. |
| negative_peak_abs_v [A] | \|min(x)\| | V | Corrected, before alignment. Recovered as \|p−V\| in the equivalent map. |
| rms_v [A] | sqrt(mean(x²)) | V | Corrected, before alignment. All 256 points. |
| mean_abs_v [A] | mean(\|x\|) | V | Corrected, before alignment. All 256 points. |
| positive_auc_vs [B] | trapz(x⁺, Δt) | V s | Corrected, before alignment. Δt = max(T,10⁻⁹ s)/255; trapezoidal rule. |
| negative_auc_abs_vs [B] | trapz(x⁻, Δt) | V s | Corrected, before alignment. x⁻=max(−x,0). |
| absolute_auc_vs [B] | positive_auc_vs + negative_auc_abs_vs | V s | Corrected, before alignment. Equivalent sum of the two integrals. |
| crest_factor [A] | max(\|x\|) / max(RMS,10⁻¹² V) | 1 | Corrected, before alignment. Peak-to-RMS ratio. |
| half_peak_width_s [C] | Δt × Σ1{\|x\| ≥ 0.5 max(\|x\|)} | s | Corrected, before alignment. Counts all threshold points, including separated regions. Zero for zero peak. |
| duration_s [C] | T = end_s − start_s | s | Historical segmentation metadata. Physical cycle duration; not 256 sample units. |
| skewness [D] | scipy.stats.skew(x, bias=False) | 1 | Corrected, before alignment. Bias-corrected third standardized moment. |
| kurtosis [D] | scipy.stats.kurtosis(x, fisher=True, bias=False) | 1 | Corrected, before alignment. Bias-corrected excess kurtosis; Gaussian reference is zero. |
| zero_crossings [D] | Σ1{signbit(xj) ≠ signbit(xj−1)} | count | Corrected, before alignment. Zero is nonnegative. No wrap at the last point. |
| template_corr [E] | Historical: corr(file template, cycle); new: max_s corr(Ts,z) | 1 | Historical metadata / training-only bank. Historical source context retained only for R01. New templates use current training cycles. |
| snr_proxy_db [E] | Historical: 20 log10(sband/snoise); new: 20 log10(RMS/sdiff) | dB | Historical pre-resampling segment / new aligned cycle. Historical value is supplied metadata. New noise is 1.4826 MAD(diff(x))/sqrt(2), floored at 10⁻¹² V. |

Historical template correlation and SNR are retained as frozen metadata for exact R01 replay. They cannot be recalculated exactly from a 256-point cycle alone. New PGT–PGD controls replace these two definitions as described below. Thus the new P16 view has the same field names but explicitly revised reference-quality calculations.

### Training-only template bank for the new controls

For each training source file s, divide each aligned cycle by its VPP and take the pointwise median, Ts. Center each template and each candidate shape by its own mean, and divide by its Euclidean norm, floored at 10⁻¹². Template correlation is the largest signed inner product over this training-only bank. No template is fitted from an outer validation or confirmation file. The SNR proxy is 20 log10[RMS(w)/snoise], with both numerator and denominator floored at 10⁻¹² V and snoise=1.4826 MAD(diff(w))/sqrt(2). This is a cycle-local proxy, not a physical instrument SNR measurement.

## Supplementary Note S3 CNN and training loss

### Table S2 Layer shapes and parameter counts

| Layer | Output shape | Settings | Parameters |
| --- | --- | --- | --- |
| Input | N×2×256 | — | 0 |
| Conv1d 2→16 | N×16×256 | k=9; stride=1; pad=4; bias | 304 |
| GroupNorm + ReLU | N×16×256 | 4 groups; ε=10⁻⁵; affine | 32 |
| MaxPool1d | N×16×128 | k=2; stride=2; pad=0 | 0 |
| Conv1d 16→32 | N×32×128 | k=5; stride=1; pad=2; bias | 2592 |
| GroupNorm + ReLU | N×32×128 | 8 groups; ε=10⁻⁵; affine | 64 |
| MaxPool1d | N×32×64 | k=2; stride=2; pad=0 | 0 |
| Global average pool | N×32 | Mean over 64 positions | 0 |
| Linear 32→8 | N×8 | Bias; no activation afterward | 264 |
| Linear 8→1 | N×1 | Bias; scalar evidence | 9 |
| Waveform subtotal | — | Original inference backbone | 3265 |
| Original training extras | N×4 logits | P6 coefficients (6), bias (1), thresholds (4) | 11 |
| Original training total | — | Waveform + P6 + ordered output | 3276 |
| New training total | — | No P6 coefficients; bias and thresholds retained | 3270 |
| Folded inference backbone | N×1 | Replace both linear layers with 32→1 | 3025 |

Convolution dilation is 1 and convolution groups are 1. Max pooling does not use ceil mode. There is no dropout. GroupNorm statistics are computed within each example; it has no running batch statistics. Final calibration coefficients are saved separately. They are not included in the CNN parameter counts.

`e = W₂(W₁h + b₁) + b₂ = (W₂W₁)h + (W₂b₁+b₂)`

The two linear layers have no intervening activation. Folding is therefore algebraically equivalent. The exported folded layer uses float64 to limit numerical differences; the convolutional backbone remains float32. Replay is checked with a 0.001 pp tolerance.

### Ordered training output

Let e be the waveform scalar. Original R01 training used a normalized P6 vector p̃, coefficients β, and bias b. The new waveform-only controls omit β and p̃. They retain the bias and ordered thresholds. The scalar exported for calibration is e, not the continuous training prediction.

`η = e + βᵀp̃ + b   (original);     η = e + b   (new controls)`

`θ₁ = t;  θk = t + Σj=1k−1[softplus(dj)+10⁻⁴],  k=2,3,4`

`qk = sigmoid(η−θk);   zk = 1{y ≥ 25k};   ŷtrain = 25 Σk=14 qk`

Initial t is −1.5. Each initial raw increment is log(exp(1)−1). The ordered thresholds ensure q1≥q2≥q3≥q4. The corresponding anchor probabilities are (1−q1, q1−q2, q2−q3, q3−q4, q4). Their expected composition is 25Σqk. This provides a continuous differentiable training output even though the supervision uses five anchors.

`L = 0.7 (1/B) Σi H₅(ŷi−yi) + 0.3 (1/4B) Σi Σk BCE(logitik,z_ik)`

`H₅(r) = 0.5r²  if |r|≤5;   H₅(r) = 5(|r|−2.5)  otherwise`

`BCE(l,z) = max(l,0) − lz + log(1+exp(−|l|))`

The target y and Huber residual are in percentage points. BCE is dimensionless. The coefficients 0.7 and 0.3 are historical numerical weights, not a scale-normalized partition of information. The epoch training log averages batch losses. Validation loss averages cycles and the four BCE thresholds. Loss weights were retained; no undocumented selection process is asserted.

## Supplementary Note S4 Final routing and calibration

The five composition anchors are a=(0,25,50,75,100). Their frozen VPP medians m are listed in Table S3. Four heads correspond to adjacent composition pairs, indexed r=0,1,2,3. The target composition is not used to select a head at inference.

### Monotonic and nonmonotonic routing

`b = ((m₀+m₁)/2, m₂, (m₃+m₄)/2)`

If every difference m(k+1)−mk has the same nonnegative or nonpositive sign, routing is monotonic. For increasing medians, r is the count of VPP≥bj. For decreasing medians, r is the count of VPP<bj. Clip r to 0–3. If all medians are equal, the increasing case is used. These boundaries are the exact historical implementation; they are not the midpoints of all neighboring medians.

`cr(V) = max(lr−V,0) + max(V−hr,0) + 0.05 |V−(lr+hr)/2| / max(range(m),10⁻⁹)`

For a nonmonotonic sequence, lr and hr are the minimum and maximum of mr and m(r+1). Choose argmin cr; an exact tie selects the first interval. The first two terms use numerical volts, and the last term is a fixed historical numerical penalty. This is an implementation rule rather than a dimensionally normalized physical distance. The frozen convention and the 0.05 coefficient are preserved and disclosed.

### Local heads and composition conversion

`ur = (y−25r)/25;   ŷseed,raw = 25r + 25 fr(X)`

R01 heads predict the local fraction u. PGD–PGS fits X=[e,P6] with StandardScaler and PLSRegression(n_components=2,scale=False). PGT–PGD fits a physical head on P16 and a joint head on [e,P16]. Each uses RobustScaler (median and interquartile range). Intervals 0 and 1 use Ridge(alpha=100); intervals 2 and 3 use linear SVR(C=100,epsilon=0.01). The local PGT–PGD fraction is 0.8 fphysical + 0.2 fjoint. The saved equivalent affine coefficients absorb the fitted centering, scaling, and final fusion.

`ŷ = clip((ŷ42,raw + ŷ123,raw + ŷ2026,raw)/3, 0, 100)`

Clipping follows seed averaging. It is not applied separately before averaging. The stored original representation concatenated 23 fields with inactive zero coefficients. The equivalent implementation retains only e plus P6 (7 fields) for PGD–PGS, or e plus P16 (17 fields) for PGT–PGD. These are alternative system paths, not 22 independent physical quantities.

### Table S3 Frozen routing statistics and calibration

| System | VPP medians at 0/25/50/75/100% (V) | Amplitude scales (V) | Heads |
| --- | --- | --- | --- |
| PGD-PGS | 4.48963594; 1.93291587; 8.01585454; 17.2914245; 25.8162369 | 42: 0.531449556<br>123: 0.531449556<br>2026: 0.531449556 | StandardScaler + PLS2 |
| PGT-PGD | 25.8162369; 21.1710601; 19.5376778; 13.9691474; 9.44341779 | 42: 1.01762652<br>123: 1.01762652<br>2026: 1.01762652 | RobustScaler; Ridge/Ridge/linear SVR/linear SVR; physical/joint=0.8/0.2 |

Supplementary Code 1 contains the full seed coefficients and calibration recipes in reference/r01/models/parameters. The weights 0.8/0.2, α=100, C=100, ε=0.01, and route penalty are historical fixed settings. Their values are recoverable, but their full selection history is unavailable. Figure S2 shows both final inference paths. The training diagram is in Supplementary Data 2, figures/cnn-training.svg (rev23-v03).

## Supplementary Note S5 Training configuration and shared controls

### Table S4 Original and new configurations

| Setting | Original R01 | New controls |
| --- | --- | --- |
| Data role | Five anchors; historical internal splits | Same frozen anchors; file-local three-block OOF |
| CNN inputs | Two channels plus normalized P6 auxiliary branch | Two channels only; one shared CNN per system/fold/seed |
| Budget | Maximum 200 epochs; 15% internal validation; patience 20 | 120 fixed epochs; no outer-block early stopping |
| Seeds | 42, 123, 2026 | 42, 123, 2026 |
| Optimizer | AdamW; lr 0.001; weight decay 0.0001; batch 64 | Same; gradient norm clipped to 1 |
| Loss | 0.7 Huber δ5 + 0.3 ordinal BCE | Same output loss; no physical branch |
| Augmentation | Shift ±4; time/amplitude 0.98–1.02; noise ≤0.005 VPP | Same bounds; saved draw ledger and deterministic random seeds |
| Sampling | Composition and source-file balance per epoch | Same balancing function; training cycles only |
| Template and SNR | Historical file-context metadata for final P16 | Training-only source templates; cycle-local SNR proxy |
| Head target | Local fraction; convert with left anchor +25u | Composition in pp directly |
| Head fitting | System-specific PLS / Ridge / SVR; different historical variants | StandardScaler + Ridge α100; no parameter search |
| Final role | Frozen R01 and historical comparisons | Separate controlled analysis; no deployment replacement |

Original early stopping used composition-equal validation MAE and required a decrease greater than 10⁻⁷ pp. Both optimizers used AdamW β=(0.9,0.999), ε=10⁻⁸, amsgrad=False, and gradient-norm clipping at 1. Original time augmentation used PyTorch bilinear interpolation with border padding and align_corners=True. New augmentation used NumPy linear interpolation with the same perturbation bounds. The interpolation implementations are not claimed to be bit-identical. Their code and random draw records are retained.

The new outer split sorts cycles within each anchor source by valid block, cycle index, and start time. numpy.array_split creates three blocks. In each fold, one block is held out and ceil(n_file^(1/3)) adjacent cycles are excluded on each available side. Shared pure-PGD cycles use identical roles in both systems. Three folds cover every anchor cycle once. The final confirmation models fit all anchors. Neither scaling nor template, route, CNN, or head fitting reads an outer evaluation label or a confirmation label. Per-file and per-fold counts are in Supplementary Data 1, sheets 12-qc-counts and 13-split-counts (rev23-v03).

Each input view has a refitted head. C is the waveform scalar; V is VPP; P is P6 or the explicitly defined new P16. C+P+V removes only exactly repeated fields. P16 already includes VPP, so it appears once. Other deterministic transformations remain. The routed view uses the same fitted global scaler as C+P+V; four Ridge heads are fitted to adjacent anchor pairs. This comparison changes local fitting and selection, not the input view or α. The head is fitted on training-CNN evidence; no cross-fitted inner representation is claimed. Outer cycles are absent from all learned fits.

OLS-V fits composition against the five training-anchor VPP medians with an intercept. Piecewise-V sorts those five medians by VPP, merges exactly equal voltages by mean composition, and linearly interpolates. Values outside the observed range use endpoint compositions. With nonmonotonic composition-to-VPP response, the inverse can be ambiguous. Sorting does not resolve physical identifiability; it defines a transparent numerical baseline. Saved JSON heads retain each fold median and interpolation order.

Protocol and input hashes were saved before training. Checkpoints, heads, and target-free predictions were locked before confirmation errors were calculated. Those errors did not trigger parameter changes, further filtering, or model selection.

## Supplementary Note S6 Evaluation reproducibility and claim limits

`MAEg = (1/ng) Σi |ŷi−yi|;   Mean MAE = (1/G) Σg MAEg;   Worst MAE = maxg MAEg`

`Biasg = mean(ŷ−y);   SD = sqrt[Σi (xi−x̄)²/(n−1)]`

Blocked OOF uses five anchor compositions; reused confirmation uses three. Mean MAE gives equal weight to compositions. Figure 5 error bars show cycle prediction SD. Seed SD describes the three individual-seed Mean MAEs; it is not experimental replicate uncertainty or an ensemble confidence interval. Supplementary Data 1 provides file, fold, seed, composition, and error-distribution metrics. Supplementary Data 2 provides all cycle predictions and errors.

For each file with n cycles, the circular block length is ceil(n^(1/3)). Random start indices are sampled; contiguous blocks wrap at the end and are concatenated to exactly n values. The same indices are used for each model in a paired comparison. File counts remain fixed. Composition MAEs are aggregated with equal composition weight. We report 20,000 replicates and percentile 95% intervals. Paired differences are test-model MAE minus reference-model MAE; negative values favor the test model. The same operation is applied to mean and worst composition MAE. Bootstrap random seeds are documented in code/audit_and_summarize.py. These are descriptive intervals for fixed models and available files, with no multiplicity-adjusted superiority claim.

Historical anchor results are anchor audits. OOF CNN evidence alone does not establish independent evaluation of the subsequent calibration. New blocked OOF excludes evaluation cycles from all learned fits, but retains historical file-context QC. The 10%, 45%, and 60% compositions are REUSED_CONFIRMATION because prior development examined them. Original acquisition-to-QC versions and parts of the historical parameter-selection record remain incomplete. Within-file cycles do not establish new-film, batch, or device generalization. The two systems were trained separately.

### Observed limits of the incremental controls

Adding P to C+V reduced blocked-OOF Mean MAE in both systems, but Piecewise-V performed better for PGD–PGS. In reused confirmation, P did not improve PGD–PGS C+V. For PGT–PGD, P reduced C+V Mean MAE but increased Worst MAE; P alone had the lowest Mean MAE. Routing reduced PGD–PGS confirmation Mean MAE but increased Worst MAE. It worsened both PGT–PGD confirmation metrics. The controls do not establish universal fusion gains or explain all sources of R01 performance.

### Earlier v02 sensitivity analysis

The earlier v02 study used a different waveform normalization and calibration protocol, so its results are reported separately. Its selected configurations passed only 1/6 reused-confirmation checks; complete results are in Supplementary Data 1, sheets 15-v02-confirmation, 19-v02-development, and 20-v02-seed-stability.

### Supplementary files and code

The public repository provides code, licenses, and a file index. Release rev23-v03 supplies Supplementary Data 1 (exact XLSX tables), Data 2 (cycle records, bootstrap results, and additional figures), and Code 1 (inputs, models, logs, and environment). Training, inference, and evaluation enter through code/reproduce.py, code/predict.py, and code/analysis.py. The release README gives download and checksum instructions. Repository release: https://github.com/YueQiandesedianzi/c02-blendpet-teng-ml-reproducibility/releases/tag/rev23-v03

## Figure S2 Final inference for the two blend systems

Figure S2. The original R01 inference paths use the saved CNN scalar e and VPP routing. PGD–PGS uses P6 with a joint PLS head. PGT–PGD uses P16 with physical and joint heads, combined with weights 0.8 and 0.2. Each seed produces a composition in percentage points. The three predictions are averaged before clipping. Note S4 and Table S3 give the route rules, five medians, and fitted-head settings. The CNN training output in Note S3 is distinct from e. The editable SVG is in Supplementary Data 2, figures/final-inference.svg (rev23-v03).

## Table S5 Data scale and exclusion summary

| Population or stage | Count | Interpretation |
| --- | --- | --- |
| Candidate physical cycles | 2,215 | Before historical QC |
| After historical QC | 2,026 | 189 cycles excluded |
| Later edge exclusions | 26 | No new filtering |
| Final unique physical cycles | 2,000 | Physical-cycle identifiers |
| PGD–PGS analysis records | 1,075 | 642 anchors; 433 reused confirmation |
| PGT–PGD analysis records | 1,152 | 731 anchors; 421 reused confirmation |
| Shared pure-PGD cycles | 227 | 454 records across the two systems |
| All analysis records | 2,227 | Includes the shared endpoint twice |

Historical QC flags overlap; their sum is not the number of excluded cycles. The 26 later exclusions were 25 cycles in 25PGT+75PGD.csv and one in 75PGD+25PGS.csv. Release rev23-v03, Supplementary Data 1, sheets 12-qc-counts, 13-split-counts, and 18-qc-reasons, retains every source and fold count. Supplementary Data 2, qc/split-ledger.csv and qc/historical-edge-filter.csv, gives cycle identities and roles. Note S5 defines the split and fit boundaries.

Overlapping QC flags comprised duration (8), VPP (34), SNR (50), baseline (12), and template correlation (160). Multiple flags can refer to one cycle. Supplementary Data 1, sheet 18-qc-reasons, retains every file and reason.

## Table S6 Model conditions

Historical conditions are descriptive. All historical rows shown in Figure 5 use VPP routing. The historical CNN comparison retains an amplitude-bearing channel and was trained with P6 supervision. Thus it is not the waveform-only C control. R04 lacks the fitted checkpoint and matching fit record; frozen predictions and metrics are retained without reconstructing missing parameters.

| History / system | Input | Amplitude / P6 training | Scaler and head | Additional handling | Checkpoint |
| --- | --- | --- | --- | --- | --- |
| R01 / PGD–PGS | e+P6 | Yes / yes | StandardScaler; PLS2 | Joint only; frozen 3 seeds | Available |
| R01 / PGT–PGD | P16 and e+P16 | Yes / yes | RobustScaler; Ridge100, linear SVR100/0.01 | 0.8 physical +0.2 joint | Available |
| R02 / both | VPP | No CNN | Imputer median; RobustScaler; Ridge100 | Local fractions | Available |
| R03 / PGD–PGS | P6 | No CNN | Imputer median; RobustScaler; Ridge100 | Local fractions | Available |
| R04 / PGT–PGD | P16 | No CNN | Not independently verified | Frozen historical metrics | Missing |
| R05 / both | CNN scalar | Yes / yes | Imputer median; RobustScaler; Ridge100 | Local fractions | Available |

Historical comparisons do not establish matched CNN budgets, hyperparameter selection, or complete calibration splits across all variants. Table S6 records verified recipes and missing checkpoints. Supplementary Data 1, sheet 10-historical-conditions, includes the full conditions matrix and R06/R07.

### New shared protocol

| Model | Inputs | Route | Head |
| --- | --- | --- | --- |
| OLS-V | 5 training medians | No | OLS with intercept |
| Piecewise-V | 5 training medians | No | Sorted interpolation; equal V merged; endpoints outside range |
| V | VPP | No | StandardScaler + Ridge100 |
| C | e | No | StandardScaler + Ridge100 |
| P | P6 / new P16 | No | StandardScaler + Ridge100 |
| C+V | e,VPP | No | StandardScaler + Ridge100 |
| C+P+V | e,P,VPP; duplicates removed | No | StandardScaler + Ridge100 |
| C+P+V+route | Same as C+P+V | Yes | Same global scaler; four adjacent-pair Ridge100 heads |

Every CNN-containing row uses the same waveform-only checkpoint for a given system, context, and seed. Each input view is refitted. All new checkpoints are available. The learned feature count is 8 for PGD–PGS C+P+V and 17 for PGT–PGD C+P+V. VPP is already one of the P16 fields.

## Table S7 Complete shared-protocol comparisons

All errors are in pp. Mean and Worst refer to composition-specific MAE. Seed SD is the sample SD of the three individual-seed Mean MAEs. Baselines have zero seed SD because they do not depend on CNN seed. Supplementary Data 1, sheets 01-summary and 04-seeds, retains seed minima, maxima, and individual results. All eight models, two systems, and two roles are retained. Supplementary Data 2, figures/composition-errors.svg, gives composition-level heatmaps (rev23-v03).

### BLOCKED_OOF

| Model | PGD–PGS<br>Mean | PGD–PGS<br>Worst | PGD–PGS<br>Seed SD | PGT–PGD<br>Mean | PGT–PGD<br>Worst | PGT–PGD<br>Seed SD |
| --- | --- | --- | --- | --- | --- | --- |
| OLS-V | 10.352 | 24.598 | 0.000 | 4.838 | 9.141 | 0.000 |
| Piecewise-V | 2.820 | 5.251 | 0.000 | 4.350 | 10.539 | 0.000 |
| V | 12.331 | 28.783 | 0.000 | 8.870 | 10.766 | 0.000 |
| C | 10.826 | 17.312 | 0.410 | 10.340 | 17.700 | 0.382 |
| P | 6.852 | 16.072 | 0.000 | 4.880 | 7.560 | 0.000 |
| C+V | 7.411 | 11.586 | 0.309 | 6.647 | 10.746 | 0.181 |
| C+P+V | 3.556 | 5.775 | 0.080 | 4.202 | 6.133 | 0.069 |
| C+P+V+route | 8.350 | 23.091 | 0.007 | 3.171 | 4.652 | 0.002 |

### REUSED_CONFIRMATION

| Model | PGD–PGS<br>Mean | PGD–PGS<br>Worst | PGD–PGS<br>Seed SD | PGT–PGD<br>Mean | PGT–PGD<br>Worst | PGT–PGD<br>Seed SD |
| --- | --- | --- | --- | --- | --- | --- |
| OLS-V | 9.026 | 11.258 | 0.000 | 10.586 | 12.757 | 0.000 |
| Piecewise-V | 7.577 | 13.575 | 0.000 | 9.282 | 14.838 | 0.000 |
| V | 10.315 | 12.263 | 0.000 | 8.394 | 11.861 | 0.000 |
| C | 8.410 | 12.275 | 4.439 | 7.356 | 11.409 | 2.388 |
| P | 7.908 | 11.678 | 0.000 | 5.128 | 11.086 | 0.000 |
| C+V | 7.052 | 13.057 | 2.325 | 8.248 | 10.942 | 1.479 |
| C+P+V | 7.217 | 10.000 | 0.421 | 7.482 | 13.739 | 0.661 |
| C+P+V+route | 6.524 | 10.610 | 0.190 | 10.482 | 21.682 | 0.033 |

## Table S7 Paired Mean MAE differences and 95% intervals

Paired differences compare Mean MAE. Negative values favor the first model. Intervals condition on the fixed models and existing files. Supplementary Data 1, sheet 07-paired-bootstrap, also reports composition-specific and Worst MAE differences.

| Comparison | OOF<br>PGD–PGS | OOF<br>PGT–PGD | Reused<br>PGD–PGS | Reused<br>PGT–PGD |
| --- | --- | --- | --- | --- |
| C+V − C | -3.415<br>[-3.610, -3.243] | -3.693<br>[-3.946, -3.438] | -1.359<br>[-1.438, -1.282] | 0.892<br>[0.587, 1.194] |
| C+P+V − C+V | -3.855<br>[-4.024, -3.656] | -2.445<br>[-2.775, -2.107] | 0.166<br>[0.022, 0.300] | -0.766<br>[-0.924, -0.609] |
| C+P+V − P | -3.297<br>[-3.462, -3.125] | -0.679<br>[-0.782, -0.577] | -0.691<br>[-0.936, -0.457] | 2.354<br>[2.097, 2.607] |
| C+P+V+route − C+P+V | 4.794<br>[4.211, 5.266] | -1.030<br>[-1.486, -0.534] | -0.693<br>[-0.888, -0.490] | 3.001<br>[2.626, 3.367] |
| C+P+V − OLS-V | -6.797<br>[-7.035, -6.521] | -0.637<br>[-0.959, -0.317] | -1.809<br>[-1.957, -1.668] | -3.104<br>[-3.305, -2.906] |
| C+P+V − Piecewise-V | 0.736<br>[0.456, 1.022] | -0.148<br>[-0.585, 0.275] | -0.360<br>[-0.674, -0.064] | -1.800<br>[-2.096, -1.508] |
