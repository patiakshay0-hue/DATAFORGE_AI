# Graph Report - D:\College\DATAFORGE_AI  (2026-08-03)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 492 nodes · 908 edges · 30 communities (29 shown, 1 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 22 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `1669fabb`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- deep_trainer.py
- dl1_routes.py
- dl1/config.py
- get
- data_routes.py
- dependencies
- ml_routes.py
- discovery.py
- report.py
- unsupervised.py
- patterns.py
- converter.py
- signals.py
- package.json
- App.jsx
- DeepLearning1View.jsx
- DeepLearningView.jsx
- useTheme
- MLView.jsx
- ChatView.jsx
- SmartConfigView.jsx
- EDAView.jsx
- DashboardView.jsx

## God Nodes (most connected - your core abstractions)
1. `get()` - 29 edges
2. `useTheme()` - 29 edges
3. `SignalBundle` - 15 edges
4. `DataProfile` - 14 edges
5. `_clamp()` - 12 edges
6. `_pattern()` - 12 edges
7. `build()` - 12 edges
8. `_clamp()` - 11 edges
9. `_pattern()` - 11 edges
10. `run()` - 10 edges

## Surprising Connections (you probably didn't know these)
- `convert_download()` --references--> `get()`  [EXTRACTED]
  backend/app/routes/convert_routes.py → backend/app/dl1_store.py
- `dl1_result()` --references--> `get()`  [EXTRACTED]
  backend/app/routes/dl1_routes.py → backend/app/dl1_store.py
- `dl1_status()` --references--> `get()`  [EXTRACTED]
  backend/app/routes/dl1_routes.py → backend/app/dl1_store.py
- `ResultsView()` --calls--> `useTheme()`  [EXTRACTED]
  frontend/src/components/MLView.jsx → frontend/src/ThemeContext.jsx
- `chat()` --calls--> `get()`  [EXTRACTED]
  backend/app/routes/chat_routes.py → backend/app/dl1_store.py

## Import Cycles
- None detected.

## Communities (30 total, 1 thin omitted)

### Community 0 - "deep_trainer.py"
Cohesion: 0.09
Nodes (46): AutoConfigRequest, ChatRequest, DeepPredictRequest, DeepSuggestRequest, DeepTrainRequest, DL1SelectRequest, Patterns the user picked from a Deep Learning 1.0 run., VisionTrainRequest (+38 more)

### Community 1 - "dl1_routes.py"
Cohesion: 0.07
Nodes (34): Any, clear(), create(), DL1Job, _evict_locked(), Per-job state for the Deep Learning 1.0 pipeline. Why this exists instead of…, Apply field updates atomically so a polling reader never sees a torn state., Test helper — drop everything. (+26 more)

### Community 2 - "dl1/config.py"
Cohesion: 0.10
Nodes (30): _activation(), _architecture(), AutoModelConfig, _batch_size(), DataProfile, _dropout(), _early_stopping(), _epochs() (+22 more)

### Community 3 - "get"
Cohesion: 0.10
Nodes (28): get(), root(), post, UploadFile, vision_get_dataset(), _vision_has_dataset(), vision_predict(), vision_status() (+20 more)

### Community 4 - "data_routes.py"
Cohesion: 0.11
Nodes (28): CleanRequest, ConvertChoice, convert_download(), convert_image_metadata(), convert_inspect(), convert_load(), convert_run(), convert_send_to_vision() (+20 more)

### Community 5 - "dependencies"
Cohesion: 0.07
Nodes (29): apexcharts, autoprefixer, axios, clsx, framer-motion, dependencies, apexcharts, autoprefixer (+21 more)

### Community 6 - "ml_routes.py"
Cohesion: 0.13
Nodes (23): Anthropic, SuggestRequest, TrainRequest, chat(), chat_status(), post, post, suggest() (+15 more)

### Community 7 - "discovery.py"
Cohesion: 0.16
Nodes (25): _anomalies(), _clamp(), _cluster_latent(), _clusters(), _compressibility(), _correlation(), _coupling(), discover() (+17 more)

### Community 8 - "report.py"
Cohesion: 0.13
Nodes (17): get_data_insights(), export_report(), ArchitectureDiagram, generate(), _header(), LossCurve, Deep Learning 1.0 PDF report. Reuses the brand styling and table helpers from…, Build the full report for a finished job. (+9 more)

### Community 9 - "unsupervised.py"
Cohesion: 0.16
Nodes (21): _activation_module(), auto_config(), AutoEncoderConfig, _build_autoencoder(), _describe(), Encoded, _intrinsic_dim(), prepare() (+13 more)

### Community 10 - "patterns.py"
Cohesion: 0.22
Nodes (20): _anomalies(), _clamp(), _clusters(), _correlation(), discover(), _feature_importance(), _influential(), _interactions() (+12 more)

### Community 11 - "converter.py"
Cohesion: 0.20
Nodes (18): _clean_parts(), convert(), _excel_sheets(), _ext(), get_converted(), get_stored_zip(), image_metadata(), inspect_upload() (+10 more)

### Community 12 - "signals.py"
Cohesion: 0.23
Nodes (18): _anomalies(), _associations(), build(), _clusters(), _correlation_structure(), _interactions(), _pca(), ndarray (+10 more)

### Community 13 - "package.json"
Cohesion: 0.11
Nodes (17): devDependencies, @types/react, @types/react-dom, vite, @vitejs/plugin-react, name, private, scripts (+9 more)

### Community 14 - "App.jsx"
Cohesion: 0.26
Nodes (8): App(), TABS, FileUpload(), formats, PreviewView(), typeMap, Ctx, ThemeProvider()

### Community 15 - "DeepLearning1View.jsx"
Cohesion: 0.17
Nodes (4): DeepLearning1View(), Progress(), STAGES, TYPE_COLORS

### Community 16 - "DeepLearningView.jsx"
Cohesion: 0.17
Nodes (5): DeepLearningView(), DEFAULT_CFG, HYPERS, LAYER_PRESETS, PERCENT_KEYS

### Community 17 - "useTheme"
Cohesion: 0.21
Nodes (6): ImageClassifierView(), FORMAT_ICON, ImportConvertView(), InsightsView(), typeConfig, useTheme()

### Community 18 - "MLView.jsx"
Cohesion: 0.22
Nodes (7): ALL_MODELS, CAT_STYLE, CATEGORIES, MLView(), MODEL_COLORS, PERCENT_KEYS, ResultsView()

### Community 19 - "ChatView.jsx"
Cohesion: 0.38
Nodes (5): ChatView(), inlineFormat(), MessageBubble(), renderText(), SUGGESTIONS

### Community 21 - "EDAView.jsx"
Cohesion: 0.40
Nodes (3): COLORS, EDAView(), METHOD_LABELS

### Community 22 - "DashboardView.jsx"
Cohesion: 0.67
Nodes (3): COLORS, DashboardView(), DbIcon()

## Knowledge Gaps
- **44 isolated node(s):** `name`, `private`, `version`, `type`, `dev` (+39 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `get()` connect `get` to `deep_trainer.py`, `dl1_routes.py`, `data_routes.py`, `ml_routes.py`, `report.py`?**
  _High betweenness centrality (0.085) - this node is a cross-community bridge._
- **Why does `infer_task()` connect `dl1/config.py` to `deep_trainer.py`?**
  _High betweenness centrality (0.076) - this node is a cross-community bridge._
- **Why does `run()` connect `dl1_routes.py` to `unsupervised.py`, `discovery.py`?**
  _High betweenness centrality (0.025) - this node is a cross-community bridge._
- **What connects `name`, `private`, `version` to the rest of the system?**
  _44 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `deep_trainer.py` be split into smaller, more focused modules?**
  _Cohesion score 0.08503401360544217 - nodes in this community are weakly interconnected._
- **Should `dl1_routes.py` be split into smaller, more focused modules?**
  _Cohesion score 0.07317073170731707 - nodes in this community are weakly interconnected._
- **Should `dl1/config.py` be split into smaller, more focused modules?**
  _Cohesion score 0.09523809523809523 - nodes in this community are weakly interconnected._