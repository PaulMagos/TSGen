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

## Level parameterisation chosen by a unit-root test

Exchange rates are integrated: the validation/test range lies outside the training range
(up to 1.77 on the train-min-max scale). In levels, the validation NLL diverges and early
stopping keeps an untrained model (prediction MAE 0.248 vs 0.0068). The fix is to feed
inputs relative to the window's first step and anchor the mixture means on the last
value, μ_m = x_t + Δμ_m (Δμ = 0 is persistence).

On stationary data the same parameterisation breaks free-running generation: the level
is invisible to the model, so samples drift like a random walk (AQI-36 MDN: 45 % of
generated values outside the training range vs 1 % in levels, with identical one-step
calibration). Feeding the absolute level as an extra input (`--level-input`) made it
worse. `--param auto` (default) therefore applies Box-Jenkins logic on the training
range only: level-relative iff most variables have a unit root (ADF, p > 0.05).
Exchange → relative (7/8 currencies), AQI-36 and Synthetic → levels. The decision and
the ADF p-values are stored in every result JSON. `--param relative|absolute` forces one.

History: AQI-36 and Synthetic were first run level-relative; `scripts/supersede_relative.sh`
moved those runs to `results/_superseded_relative/` (ASGTM kept as the `asgtm-relative`
ablation) and `scripts/rerun_auto_param.sh` reran them. In MLflow the old runs carry the
tag `superseded`.

## Ablations wired into the grid

`--temporal-graph chain|complete|hvg|none` (A3), `--edge-weight similarity`,
`--spatial-graph random` (A4), `--param relative|absolute` (A5). See `scripts/run_grid.sh`.
`scripts/thesis_tables.py` writes the thesis tables with Welch significance marks and
`comparisons.md` (every research-question contrast).

## Tracking with MLflow

```bash
infra/mlflow/up.sh      # MLflow 3.16.1 + Postgres in Docker (ArcBox); random DB password in infra/mlflow/.env
export MLFLOW_TRACKING_URI=http://mlflow.tsgen-mlflow.arcbox.local:5000
bash scripts/run_grid.sh                       # every run is logged live
.venv/bin/python -m tsgen.tracking backfill    # log results/ JSONs produced without tracking
```

One experiment per dataset (`tsgen/AirQuality`, …), one run per model × seed, with the
config as params, `train_loss`/`val_loss` per epoch, every final score
(`prediction/mae`, `generation/vs_train/acf_distance`, …), CPU/RAM/GPU system metrics
and the result JSON as artifact. Without `MLFLOW_TRACKING_URI` tracking is a no-op.

The server publishes no host port (ArcBox's `127.0.0.1:PORT` forwarding is broken and a
plain port would expose an unauthenticated server to the LAN). A remote GPU machine logs
through an SSH reverse tunnel opened from this Mac:

```bash
ssh -R 5050:mlflow.tsgen-mlflow.arcbox.local:5000 user@gpu-box
# on gpu-box:
export MLFLOW_TRACKING_URI=http://localhost:5050
DATASETS=airquality DEVICE=cuda bash scripts/run_grid.sh
```

## Capacity sweep

`scripts/run_capacity.sh` reruns MDN / GTM / ASGTM with budgets set by `--hidden 128` and
`256` (tags `-h128`, `-h256`; the main grid is hidden 64). For AQI-36 the budgets are
≈0.25M / 0.53M / 1.19M parameters; the legacy ASGTM had ≈0.72M.

## Not yet included (phase 4)

DGAN / PAR / TimeGAN baselines, GRIN on the same AQI split, and the GRIN month-based
AQI split (this core uses a chronological split for every dataset).
