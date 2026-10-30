"""Reproduce real-source validation without creating history entries."""
from pathlib import Path
import json
import sys
import importlib.metadata
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ingestion.stanford_parser import parse_hpr
from src.preprocessing.feature_engineering import prepare
from src.ml.regression import Config, ALGORITHMS, run_experiment

root = Path(__file__).resolve().parents[1]
dataset = parse_hpr(next(root.glob('25119*.txt')), recover=True)
trees, data, invalid = prepare(dataset)
runs = []
for target, features in [('LogLength', ['DBH', 'SpeciesGroupKey', 'ProductKey', 'ButtDiameter_ob']), ('Volume_price', ['LogLength', 'ButtDiameter_ob', 'TopDiameter_ob', 'DBH', 'ProductKey']), ('CuttingDuration', ['LogLength', 'MidDiameter_ob', 'DBH', 'CuttingReason'])]:
    for algorithm in ALGORITHMS:
        result = run_experiment(data, Config(target, features, algorithm=algorithm))
        runs.append(result['metadata'])
        print(target, algorithm, result['metadata']['test'], flush=True)
report = {'source_sha256': dataset.fingerprint, 'ingestion': dataset.report, 'analytic_exclusions': invalid,
          'recorded_price_volume_m3': float(data.Volume_price.sum(min_count=1)),
          'missing_logscom_ids': int(data.LogscomID.isna().sum()),
          'cutting_reasons': data.CuttingReason.value_counts().to_dict(),
          'versions': {x: importlib.metadata.version(x) for x in ['streamlit', 'pandas', 'numpy', 'scikit-learn', 'shap', 'lxml']},
          'runs': runs}
(root / 'artifacts').mkdir(exist_ok=True)
(root / 'artifacts/real_data_validation.json').write_text(json.dumps(report, indent=2, allow_nan=False))
