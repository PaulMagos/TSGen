# tsgen — fixed methodology core

A clean re-implementation of the thesis model family (MDN → GTM → SGTM → ASGTM) that
matches the formulation in the Methods chapter and fixes every issue in
`../THESIS_REVIEW.md` §1–2 and `../METHODOLOGY_REVIEW.md` §3. The legacy code (`GT/`,
`TrainScript*.py`) is untouched, so the submitted numbers stay traceable.

```bash
uv venv -p 3.11 .venv && uv pip install -p .venv/bin/python -r requirements-core.txt
.venv/bin/python -m pytest -q                      # 56 tests, ~4 s
.venv/bin/python run_experiment.py --dataset airquality --model asgtm --seed 0
SEEDS="0 1 2 3 4" DEVICE=cuda bash scripts/run_grid.sh   # full grid + ablations
.venv/bin/python aggregate.py --latex results/tables.tex
```

Data is downloaded on first use into `data_cache/` (Exchange: Lai et al. 2018; AQI-36:
Zheng et al. 2015 with the standard `eval_mask`). Synthetic follows
`GenerateSyntheticDataset.ipynb`.

## What changed, and where

| Review finding | Fix | Code | Test |
|---|---|---|---|
| Loss = mean over components of −log(π·L) | exact mixture NLL, logsumexp, D-dim normaliser | `models.MixtureHead.nll` | `test_nll_equals_torch_mixture` |
| Sampling = weighted sum Σπ_mY_m | k ~ Cat(π), x ~ N(μ_k, diag σ_k²) | `MixtureHead.sample` | `test_sampling_is_component_sampling_not_weighted_sum`, `test_mixture_recovers_two_modes` |
| σ was a variance, shared across nodes | per-node σ ∈ R^{M×N}, softplus | `MixtureHead` | — |
| No split, scaler on all data | chronological 70/10/20, scaler on train | `data.build_series` | `test_splits_are_chronological_and_scaler_uses_train_only` |
| Non-overlapping windows (35–52 samples) | stride-1 windows; val/test windows use past context only | `data.window_starts` | `test_windows_do_not_cross_into_later_splits` |
| Bidirectional LSTM, backward diffusion, window-as-channel spatial conv | unidirectional LSTM, causal temporal diffusion, per-step spatial diffusion | `models.*` | `test_outputs_before_t_ignore_the_future` (4 variants × 3 graphs) |
| VG not causal | causal (vector) VG via RustyGraph; exact prefix property | `graphs.py` | `test_visibility_prefix_property`, `test_temporal_graphs_are_strictly_causal` |
| Edge weight = distance (dissimilar ⇒ bigger message) | `binary` default, `similarity` option | `graphs.temporal_adjacency` | — |
| Exo columns treated as graph nodes (adjacency off by 2) | spatial block sees sensors only | `models.SpatialDiffusion` | `test_spatial_messages_follow_the_given_adjacency` |
| Adaptive graph always dense | top-k sparse softmax | `models.AdaptiveAdjacency` | `test_adaptive_adjacency_is_sparse_and_stochastic` |
| Imputation fed the true values | p(x_t \| observed past, same-step observed entries): hidden inputs presented as in training (last observation + mask flag), no feedback of imputed values (feedback compounds the anchored means: AQI MAE 0.11 → 0.06); held-out entries hidden in training too | `inference.impute`, `data.with_eval_mask` | `test_imputation_never_reads_hidden_values` |
| AQI scored on tsl-filled values | scored on the real `eval_mask` observations | `data.load_aqi` | — |
| Generation resampled the whole window, noise seed, no burn-in | append one step, real train seed, burn-in discarded | `inference.generate` | `test_generation_shapes_and_burn_in` |
| Early stopping kept references, monitored train loss | deep copy, validation loss | `train.fit` | `test_early_stopping_restores_best_weights` |
| Baselines trained on windows, evaluated on full history; unequal sizes | same windows for all; parameter-matched to ASGTM | `run_experiment.make_model` | — |
| Metric arg order, MASE on node axis, JSD distance, p-values averaged, biased MMD | fixed; + CRPS, ACF / cross-corr distance, discriminative score, TSTR, memorisation, visibility-graph fidelity | `metrics.py` | `tests/test_metrics.py` (30) |
| No sanity baselines | persistence, linear extrapolation, VAR(AIC), LOCF, interpolation | `baselines.py` | `test_var_recovers_var1` |
| Exchange exo constant (sin 0, cos 1) | no calendar covariate for Exchange (file has no real dates); AQI: hour + weekday | `data.py` | — |

## New modelling choice: level-relative parameterisation

Exchange rates are non-stationary: the validation/test range lies outside the training
range (up to 1.77 on the train-min-max scale). With absolute inputs and means the
validation NLL diverges from the first epoch and early stopping keeps an untrained model
(MAE 0.23 vs 0.006 for persistence). The core therefore feeds inputs relative to the
window's first step and anchors the mixture means at the last value,
μ_m = x_t + Δμ_m (persistence is the Δμ = 0 model). `--absolute` restores the original
parameterisation as an ablation.

## Ablations wired into the grid

`--temporal-graph chain|complete|hvg|none` (A3), `--edge-weight similarity`,
`--spatial-graph random` (A4), `--absolute` (A5). See `scripts/run_grid.sh`.

## Not yet included (phase 4)

DGAN / PAR / TimeGAN baselines, GRIN on the same AQI split, and the GRIN month-based
AQI split (this core uses a chronological split for every dataset).
