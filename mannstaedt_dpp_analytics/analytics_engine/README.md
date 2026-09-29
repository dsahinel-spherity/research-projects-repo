# analytics_engine — starter kit

A trimmed, self-contained copy of Spherity's `danalyticAPI` pattern
(originally built for football data analysis), adapted as a template for
`mannstaedt_dpp_analytics`. The full explanation — especially how
`AnalyticEngine` works — is in the Notion ticket Doruk shared alongside
this folder. This README is the quick-reference version.

## What's in here

```
analytics_engine/
├── __init__.py
├── data_utils.py          # Logger, df<->dict/list helpers, generic file readers
├── engine.py               # AnalyticEngine — the single entry point
├── example_usage.py        # runnable demo (python -m analytics_engine.example_usage)
├── requirements.txt
└── algorithms/
    ├── __init__.py
    ├── alg.py               # every setup_data key, in one place
    ├── algorithm.py         # base Algorithm class (data loading + column handling)
    ├── cluster.py            # KMeans + elbow/silhouette + PCA visualisation
    ├── feature.py             # feature selection: PCA, correlation, ANOVA, mutual info, Lasso, chi-square
    ├── linearreg.py           # linear regression
    └── logisticreg.py         # logistic regression
```

## What was left out of the original project, and why

The original `danalyticAPI` also had `dtree.py`, `gradboostreg.py` and
`svm.py`. None of those made the cut:

- `dtree.py` never followed the shared `Algorithm` base class, hardcoded
  file paths, and pulled in a `graphviz`/`pydotplus` plotting dependency
  for a side effect. It also wasn't registered in `engine.py`'s dispatch
  table, so it was never reachable through the engine anyway.
- `gradboostreg.py` and `svm.py` were bare top-level scripts (not
  classes) with hardcoded filenames like `"dtfirst.xlsx"` and stray
  `plt.show()` calls — leftover exploration notebooks, not part of the
  working pattern. `svm` was in fact explicitly mapped to `None` in the
  original `engine.py`, i.e. "not implemented."

The original project's `load_data_from_db` also called a proprietary
football-match data feed (`GameAPI`). That's specific to the football
analytics contract and has no bearing on Mannstaedt's web-traffic/DPP
data, so it's been replaced with a stub in `algorithms/algorithm.py` —
see the docstring there for how to point it at, e.g., the existing
`app/services/umami_service.py` in this repo instead. `load_data_from_file`
and `load_data_from_url` are untouched and work as-is; they're already
generic pandas/requests code.

Everything else here is logically unchanged from the original — only
import paths were updated (the original imported from `src.danalyticAPI...`
and `src.utils.Utils`, neither of which exists in this standalone copy;
`data_utils.py` inlines the small handful of generic helpers those modules
provided) and one small bug was fixed (`engine.py`'s `_get_algorithm`
referenced an undefined `name` variable in its error log).

## Try it

```bash
pip install -r requirements.txt
python -m analytics_engine.example_usage
```

That fits a linear regression on a small synthetic CSV entirely through
`AnalyticEngine` — no algorithm code touched directly.

## AnalyticEngine, in short

`AnalyticEngine(setup_data).handle()` does four things, in order:

1. **Pick an algorithm** — reads `setup_data["algorithm"]["name"]`
   (e.g. `"linear-regression"`), looks it up in the `functions` dict at
   the top of `engine.py`, and instantiates that class.
2. **Pick a method** — reads `setup_data["algorithm"]["function"]`
   (e.g. `"predict"`, `"train"`, or any public method the class defines,
   like `Cluster.elbow` or `FeatureSelection.correlation`) and checks
   the chosen algorithm actually has it.
3. **Load and validate input** — calls the algorithm's `setup(setup_data)`
   (which loads data via `setup_data["data"]` and applies
   `setup_data["features"]`/`setup_data["algorithm"]["config"]`), then
   checks `setup_data["algorithm"]["input"]` only contains keys the
   chosen method actually accepts (via `inspect.signature`).
4. **Run it** — calls the method with that input and returns whatever it
   returns (a dict, a list of rows, a score — whatever that method
   produces).

The point of this indirection: adding a new algorithm to the whole system
never touches `engine.py`'s logic, only its `functions` dict (one line),
plus a new file under `algorithms/` that subclasses `Algorithm`. The
caller (a REST endpoint, a Streamlit callback, a test) only ever talks in
`setup_data` dicts — it never imports or instantiates an algorithm class
directly.

## Wiring this into `mannstaedt_dpp_analytics`

This repo currently runs as a single Streamlit container (see the root
`Dockerfile`) with no `docker-compose.yml` yet. Two reasonable ways to add
this engine:

**A. In-process** — copy `analytics_engine/` under `app/`, import
`AnalyticEngine` directly from `app/services/ml_service.py`, and swap its
existing ad hoc sklearn code for `setup_data` calls into the engine. No
Docker/compose changes needed at all — everything still runs in the one
existing container.

**B. Separate service** — wrap `AnalyticEngine` in a small FastAPI app
(the `requirements.txt` in `app/` already lists `fastapi`/`uvicorn`,
currently unused) exposing something like `POST /run` that accepts a
`setup_data` JSON body. Give it its own `Dockerfile` (can reuse this
folder's `requirements.txt` plus the FastAPI/uvicorn lines) and add a new
`docker-compose.yml` with two services — the existing `dashboard`
(Streamlit) and a new `analytics-engine` (FastAPI) — on a shared network,
with the dashboard calling the engine over HTTP instead of importing it.

Either way, the first real integration step is filling in
`algorithms/algorithm.py`'s `load_data_from_db` stub (or just using
`load_data_from_file`/`load_data_from_url`, which already work) so the
engine can pull Mannstaedt's actual traffic/DPP data instead of the demo
CSV in `example_usage.py`.
