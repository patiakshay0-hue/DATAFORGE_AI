# Deep Learning 2.0 — Implementation Plan

## Decisions taken

1. **DL 1.0 supersedes the existing Deep Learning tab** — its charts and prediction
   playground migrate into the new pipeline. One nav entry, not two.
2. **Report extends the existing reportlab PDF** (`generate_pdf_report`), charts drawn
   natively as reportlab vectors. Server-side and reproducible.
3. **Backend first (Phases 1–2)**, verified by curl before any UI is written.

## Status

- **Phase 1 — Backend foundation: DONE.** `app/dl1_store.py`, `core/dl1/config.py`
  (`AutoModelConfig`), plus activation / optimizer / loss / early-stopping wired
  through both the torch and scikit-learn training paths.
- **Phase 2 — Pattern engine: DONE.** `core/dl1/signals.py` (`SignalBundle`),
  `core/dl1/patterns.py` (nine detectors).
- **Phase 3–5 — pipeline, routes, frontend, report: NOT STARTED.**

Two bugs were found and fixed while building:
`infer_task` (integer class labels with >10 distinct values were misread as
regression — now shared by `_prepare_rich` so config and training agree), and
cluster-count parsimony (a 30-row dataset was reporting 8 clusters).

## Headline finding

**Roughly 65% of this spec already exists in the codebase.** The naive reading of the
prompt ("build a new module") would duplicate `deep_trainer.py` (657 lines), the
training loop, permutation importance, evaluation, and the PDF report engine.

The optimized path is: **reuse the proven pipeline, and spend the effort on the four
things that genuinely don't exist yet** — full hyperparameter selection, the pattern
_narrative_ layer, pattern selection, and the extended report.

---

## 1. Reuse map — do NOT rebuild these

| Spec section                                                                    | Already implemented                                   | Location                                           |
| ------------------------------------------------------------------------------- | ----------------------------------------------------- | -------------------------------------------------- |
| §2 Auto preprocessing (numeric/categorical detection, missing values, encoding) | `_prepare_rich()`                                     | `backend/core/deep_trainer.py:85`                  |
| §2 Target column detection & ranking                                            | `recommend_targets()`                                 | `deep_trainer.py:170`                              |
| §4 Feature importance                                                           | `_permutation_importance()`                           | `deep_trainer.py:268`                              |
| §4 Correlation / PCA / clusters / feature ranking                               | `discover_patterns()`                                 | `deep_trainer.py:581`                              |
| §6 Confusion matrix, ROC, predicted-vs-actual                                   | `_evaluation()`                                       | `deep_trainer.py:290`                              |
| §6 Accuracy/Precision/Recall/F1 · RMSE/MAE/R²                                   | `_final_metrics()`                                    | `deep_trainer.py:241`                              |
| §6 PDF report engine + brand styling                                            | `generate_pdf_report()`                               | `backend/core/exporter.py:78`                      |
| §6 Loss/accuracy curves, confusion matrix UI                                    | `ResultsView`, `ClassificationEval`, `RegressionEval` | `frontend/src/components/DeepLearningView.jsx:280` |
| §7 Progress indicator during training                                           | `TrainingScreen`                                      | `DeepLearningView.jsx:256`                         |
| §7 Streaming progress transport                                                 | SSE via `StreamingResponse`                           | `backend/app/routes/chat_routes.py:27`             |
| §1 Nav pattern (`TABS` + `alwaysOn` + render switch)                            | —                                                     | `frontend/src/App.jsx:35`                          |

**Consequence:** the training/evaluation core is done and working (verified: a real
`/deep/train` run returns all 13 fields the UI consumes). Treat it as a dependency,
not as something to fork.

---

## 2. The three optimizations that cut the most work

### O1 — One job + SSE progress, not six round-trips

Spec §7 defines an 8-stage pipeline, each stage needing a progress indicator. The
obvious implementation is 6 sequential POSTs from the browser, each re-reading global
state and recomputing overlapping work.

Instead:

```
POST /dl1/run          -> { job_id }        (kicks off, returns immediately)
GET  /dl1/stream/{id}  -> SSE event stream  (stage, pct, message)
GET  /dl1/result/{id}  -> full payload
```

One dataframe pass. Patterns computed once. The SSE transport is already proven in
this repo by `chat_routes.py` — copy that shape, don't invent one.

**Saves:** ~5 redundant preprocessing passes per run, and gives §7's progress display
for free instead of the current fake `setInterval` timer
(`DeepLearningView.jsx:71` currently _simulates_ progress with a 1-second tick).

### O2 — Per-job store, not the global `data_store`

`backend/app/store.py` is literally `data_store = {}` — one module-level dict shared
by every request. `deep_trainer.predict()` likewise leans on a module-global for the
last trained model.

DL 1.0 makes this worse: the user trains, _then browses patterns, then selects one,
then generates a report_ — minutes apart. Any second upload in another tab clobbers
the run mid-flow.

Introduce `backend/app/dl1_store.py` with `jobs: dict[str, DL1Job]` + TTL eviction.
This is also the clean extension point for "Deep Learning 2.0" required by §8.

### O3 — Compute the signal once, narrate it nine times

§4 asks for nine pattern categories (importance, correlation, clusters, non-linear
relationships, interactions, influential variables, segments, anomalies,
target-drivers). Running a separate analysis per category is the expensive mistake —
`discover_patterns()` alone already runs KMeans for k=2..10 _plus_ silhouette scoring.

Structure it as **one heavy pass → one shared signal bundle → nine cheap interpreters**:

```
SignalBundle = { corr_matrix, pca, cluster_labels+centroids,
                 permutation_importance, variance_rank, residuals }

PatternDetector.detect(bundle) -> [Pattern, Pattern, ...]
```

Each detector is then a pure function over already-computed numbers. Turns
`O(9 × heavy)` into `O(1 × heavy + 9 × trivial)`.

---

## 3. Gap analysis — what actually needs building

| #   | Spec                    | Status      | Work                                                                                                                                                                                                        |
| --- | ----------------------- | ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| A1  | §3 `AutoModelConfig`    | **Partial** | `auto_optimize_config()` returns `hidden_layers, epochs, learning_rate, dropout, batch_size`. **Missing: `optimizer`, `activation`, `loss_function`, `early_stopping`**                                     |
| A2  | §3 applying that config | **Partial** | Optimizer is hardcoded Adam (`deep_trainer.py:355`), activation hardcoded ReLU (`:319`), loss hardcoded (`:347,351`), **early stopping does not exist** — the loop runs all epochs unconditionally (`:370`) |
| A3  | §4 Pattern narrative    | **Missing** | `discover_patterns()` returns raw numbers. Spec needs `{title, description, confidence, columns, visualization}` per pattern                                                                                |
| A4  | §5 Pattern selection    | **Missing** | No select / compare / mark-preferred state anywhere                                                                                                                                                         |
| A5  | §6 Extended report      | **Partial** | PDF engine exists but covers schema/EDA/insights only. Needs model config, selected pattern, features used vs ignored, feature ranking, DL metrics, DL charts                                               |
| A6  | §1 Navigation           | **Trivial** | One `TABS` entry + one render line in `App.jsx`                                                                                                                                                             |

---

## 4. Build phases

### Phase 1 — Backend foundation

- `backend/app/dl1_store.py` — job store, TTL, status enum
- `backend/core/dl1/config.py` — `AutoModelConfig` class (A1). Extends the existing
  heuristics with optimizer / activation / loss / early-stopping rules driven by
  dataset size, feature count, task type, class balance
- Extend `_build_torch_mlp()` + `_train_torch()` to honour activation, optimizer and
  patience-based early stopping with best-weight restore (A2)

### Phase 2 — Pattern engine

- `backend/core/dl1/signals.py` — one pass producing `SignalBundle` (O3), fed by the
  **encoded** matrix from `_prepare_rich()` rather than raw numerics (see Risk R2)
- `backend/core/dl1/patterns.py` — nine detectors + confidence scoring (A3)

### Phase 3 — Orchestration

- `backend/core/dl1/pipeline.py` — the 8-stage runner, yielding progress events
- `backend/app/routes/dl1_routes.py` — `run` / `stream` / `result` / `select` / `report`

### Phase 4 — Frontend

- `frontend/src/components/dl1/` — `DL1View`, `StageProgress`, `PatternBrowser`,
  `PatternCompare`, `ReportView`
- Nav entry + render line in `App.jsx` (A6)

### Phase 5 — Report

- `backend/core/dl1/report.py` — extends `generate_pdf_report()` sections (A5)

**Order matters:** Phases 1–2 are independently testable via curl before any UI
exists. Build and verify them first; the frontend is then a thin consumer.

---

## 5. Risks / decisions

**R1 — torch is not in `requirements.txt`.** Locally you have torch 2.13.0+cpu, but
lines 20–21 of `requirements.txt` deliberately comment it out (Render free-tier size
limit). `deep_trainer` already falls back to scikit-learn. DL 1.0's new
optimizer/activation/early-stopping choices **must degrade gracefully on the sklearn
path** or the module breaks in production. sklearn's `MLPClassifier` does support
`solver`, `activation` and `early_stopping`, so a mapping exists — but it is lossy
(no per-epoch history control). Plan for a documented capability matrix.

**R2 — `discover_patterns()` is numeric-only.** Line 582 does
`select_dtypes(include=[np.number])`, so categorical columns are dropped entirely.
Spec §4 wants "data segments" and "anomaly groups", which are usually _driven_ by
categoricals. Fix by feeding it the encoded matrix from `_prepare_rich()`.

**R3 — Correlation heatmap has no Recharts primitive.** Recharts (already used)
can't do heatmaps. `apexcharts` + `react-apexcharts` are now installed and have a
native heatmap — recommend using those here rather than hand-rolling an SVG grid.

**R4 — Charts in the PDF.** reportlab cannot render Recharts output. Three options:
(a) server-side matplotlib → PNG → embed (adds a dependency);
(b) client captures chart canvases and posts PNGs back;
(c) draw the architecture diagram and heatmap natively with reportlab vector calls.
Recommend (c) — both shapes are simple, and it keeps report generation server-side
and reproducible.

**R5 — Existing "Deep Learning" tab.** There is already a working Deep Learning
section. Shipping "Deep Learning 2.0" alongside it means two similar nav items.
Decide: coexist, or supersede.
