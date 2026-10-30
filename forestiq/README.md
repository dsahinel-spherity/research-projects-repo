# ForestIQ — Timber Analytics

A local Streamlit application for StanForD 2010 HPR timber production. The beginner interface has four pages, the supplied LogChain logo and no emoji navigation icons:

- **Overview:** trees, logs, recorded volume, estimated harvest area and a map.
- **Timber Data:** a per-log table, felling/drop-off provenance, local dataset registration, CSV and GeoJSON downloads.
- **Log Search & Traceability:** search by LogscomID, tree, log or product; inspect recorded locations and download a selected-log GeoJSON.
- **Analytics:** production by time/product, log-length distributions, missing-data coverage and an optional unusual-measurement check.

The sidebar's **Show advanced tools** switch exposes the earlier data exploration, regression, explainability, anomaly detection, clustering and experiment history pages. They are hidden by default.

## Run

Python 3.11 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

In the already prepared workspace:

```bash
.venv/bin/python -m streamlit run app.py
```

The repository includes `25119_KL_Axertal_Wagenburg_Logscom_erklaert_NEU (2).txt` unchanged. On a fresh checkout, the app opens it as **Bundled sample**, automatically enables complete-tree recovery and prepares the timber tables and harvest footprint. No upload or generated registry files are needed to start.

To also register the sample locally before starting the app, run:

```bash
python -m scripts.initialize_sample
```

Open http://localhost:8501. No paid API or external LLM is used. Map basemap tiles require internet; processing, exports and area calculations run locally.

## Timber datasets

Select a registered dataset in the sidebar or upload a complete `.hpr`, `.xml` or XML-based `.txt` file. An active upload takes precedence until you clear it or choose another dataset. Dataset selection, upload bytes, recovery settings and filters persist across sections within the browser session. They are not retained after the session ends unless the dataset is registered. Files are parsed in full, without sampling or truncating records; the size limit is 200 MB. A file in `data/raw/` can replace the bundled default.

**Register this timber dataset** saves the complete source and derived tables under `data/registry/<content-id>/`. Registration always uses the full source, while on-screen CSV and GeoJSON downloads use the current filters. Registrations are idempotent for the same source, recovery setting and preparation version.

Each manifest identifies:

- `dataset_type: timber`
- `preparation_branch: timber_hpr_v1`
- Source SHA-256, registration time, schema/relational validation, data exclusions and recovery setting.

After registration, Timber Data immediately shows whether the official HPR schema check passed or failed, with issue details on failure. Recovered incomplete files are marked as damaged/incomplete; unsupported versions are marked as not verified. Offline checks cover HPR 2.1, 3.3, 3.6 and 4.1. Saving a dataset does not mean it passed validation, and passing the schema check does not guarantee measurement accuracy. The result stays visible during the session for the source just registered.
- `source.hpr`: exact original bytes, including all headers, definitions and unrecognized extensions.
- `trees.csv`: one row per machine/stem identity, including trees without logs.
- `timber_logs.csv`: one row per machine/stem/log identity.
- `harvest_area.geojson`: full registered dataset's estimated felling-point footprint.

The registry has its own explicit preparation entry point in `src/preprocessing/timber.py`; it does not route timber records through a generic industrial-table preparation routine. Machine-scoped IDs prevent collisions between trees and product definitions from different machines. Product name and user ID are resolved from each machine's definitions when present.

## Per-log event mapping

### Search and traceability

Open **Log Search & Traceability**, choose `LogscomID`, and enter `105` to find the sample's tree `128373`, log `1`. Search is exact after trimming whitespace; the matching-log dropdown is also searchable. An empty query allows browsing all logs in the current sidebar scope. Duplicate identifiers require explicit record selection. Missing LogscomID values can be found through StemKey, LogKey or ProductKey. Internal selection identity includes the dataset fingerprint, machine, StemKey and LogKey.

The PyDeck map shows green felling markers and blue **recorded product positions**, with recorded timestamps in tooltips and the information panel. The optional **Link between recorded points** is off by default. Its dashed visual connection is not a measured transportation route. All four layers have independent visibility controls. The map fits the selected positions; **Zoom to harvest footprint** shows the broader area. Coincident positions use concentric markers and no connection line. Missing or out-of-range positions are omitted, not substituted.

`log_<LogscomID>_traceability.geojson` contains the supported selected-log Points, a straight LineString when positions differ, and the current estimated footprint Polygon when available. If LogscomID is absent, the filename uses StemKey and LogKey. Exports always use `[longitude, latitude]` and valid JSON with null rather than NaN for missing properties. Layer toggles control display only; the export retains all available supported geometry. The footprint contains only the current scope's trees. Inside/on-boundary/outside status is largely a consistency check because the hull comes from those same points, not an independent boundary survey.

Implementation: `src/utils/traceability.py` handles search, geometry and export; `src/traceability_ui.py` builds PyDeck layers; `pages/12_Log_Search.py` integrates the workflow. Tests in `tests/test_traceability.py` cover exact and ambiguous search, identity scope, coordinate order, GeoJSON, missing positions, containment, layer toggles and the ID-105 user workflow. No extra geospatial dependencies are required beyond the existing PyProj/SciPy/PyDeck stack.

### Event sources

| Output | Source and meaning |
|---|---|
| Tree | MachineKey + StemKey; original StemKey and StemNumber retained |
| Log | TreeID + LogKey; LogscomID retained when available |
| Product | ProductKey, ProductName and ProductUserID where defined |
| Volume | Price, over-bark and under-bark volumes remain separate, in m³ |
| Felling point | `StemCoordinates` at crane tip when felling (3.x) or attachment center (4.x) |
| Felling time | `FellCutStartTime`; otherwise `HarvestDate`, explicitly identified in `FellingTimeSource` |
| Drop-off point | `LogCoordinates` with receiver position `Product position` |
| Drop-off time | Product-position `CoordinateDate` |

A product-position timestamp is the time of the coordinate observation, not an independently verified release event. `CuttingEndTime` is **not** substituted for missing drop-off time. Machine-base coordinates are **not** substituted for felling points. Unavailable data stays blank. Positions are exported only when valid WGS84 / EPSG:4326 coordinates are declared at the coordinate or document level; recognized vendor EPSG extension codes are supported. Other CRS values are excluded from geographic outputs rather than guessed.

## Harvest footprint / GeoJSON

The hull uses distinct recorded felling points from trees in the current scope. Coordinates are projected to a local Lambert azimuthal equal-area system for hull construction; area and perimeter are measured geodesically on the WGS84 ellipsoid with `pyproj.Geod`. GeoJSON coordinates are longitude, latitude, with a closed counterclockwise exterior ring. No arbitrary buffer is added.

The supplied sample produces **1.2440536419 ha** from 56 felling points, approximately **1.24 ha**. This is an estimated footprint, not a surveyed boundary: a convex hull includes gaps between trees. Missing points are reported. Fewer than three distinct non-collinear points cannot produce a polygon. Widely separated operations and antimeridian-spanning points are rejected; select a single harvest object using the sidebar. The area is computed from data, never forced to the sample's expected number.

Convenience exports of the registered sample are in `artifacts/harvest_area.geojson` and `artifacts/timber_logs.csv`.

## Complete HPR support and validation boundaries

The parser reads complete `HarvestedProduction` documents in `urn:skogforsk:stanford2010`, including prefixed XML, multiple machines, definitions, single/multi-tree processing records, felled-only trees, nested log measurements and optional/vendor event fields. Modern 4.x and older 3.x receiver-position labels are recognized. Standard direct log coordinates and the sample's extension-nested coordinates are handled. Unrecognized content is retained losslessly in the registered source rather than asserted to have known analytical semantics.

Offline **official XSD checks for 2.1, 3.3, 3.6 and 4.1** are included. The report distinguishes valid, invalid, recovered and unchecked-version files. Other versions receive structural extraction and explicit unit checks, not a claim of XSD certification. Full extraction does not imply every vendor extension is semantically supported. Ordinary HPR files may lack precise felling/drop-off events entirely; no engine can reconstruct those measurements from missing data. This is not a parser for the legacy pre-2010 non-XML StanForD format or other message types such as FPR.

The normalized measurements currently require explicit mm, cm and m3 units. XML entity/DTD declarations and external network resolution are disabled. Duplicate or missing relational IDs block analytics. Invalid measurements remain in raw parsed tables and are excluded only from analytical copies. Ambiguous duplicate coordinates/measurements are left missing and reported.

### Supplied sample caveat

The original `.txt` is not an intact HPR document. It contains a German legend, a damaged opening section and two orphan log-key occurrences. Explicit recovery keeps 56 complete trees and 323 logs, including 245 automatic and 78 other-manual cuts. No tree identity is invented. Recovery is enabled for the bundled sample and disabled for new uploads; it only supports intact unprefixed tree blocks in a single-machine document. The dashboard keeps the recovery warning visible. Totals and area describe the recovered records only. The original file is unchanged.

## Advanced tools

The retained modules provide four regression algorithms, tree-grouped holdout/CV, optional forward-time evaluation, baseline comparisons, permutation importance/SHAP, Isolation Forest, K-Means/DBSCAN/PCA and descriptive statistics. Preprocessing is fit only on training folds. Pre-cut features require confirmation of acquisition timing. Saved JSON experiments remain local; models stay in session memory. Anomaly flags are not verified machine faults; SHAP associations are not causal claims.

## Verification

```bash
python -m pytest -q
python scripts/validate_real_data.py
```

Tests cover real source ingestion, units, missing/duplicate IDs, machine scoping, leakage, advanced models and explanations, all page renders, beginner/advanced navigation, timber registration, provenance, modern coordinate labels, felled-only files and polygon export. XML test mutations exercise schema behavior without substituting synthetic production data for the real sample. The regression-validation script writes actual held-out metrics to `artifacts/real_data_validation.json`.

Official references: [Skogforsk StanForD releases](https://www.skogforsk.se/english/projects/stanford/stanford-2010/), [scikit-learn group validation](https://scikit-learn.org/stable/modules/cross_validation.html), [SHAP API](https://shap.readthedocs.io/en/stable/api.html). Logo provenance is recorded in `assets/README.md`.
