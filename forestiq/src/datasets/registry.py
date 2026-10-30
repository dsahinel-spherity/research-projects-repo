"""Local, content-addressed timber dataset registry with a dedicated preparation branch."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import tempfile
from src.ingestion.stanford_parser import parse_hpr
from src.preprocessing.timber import BRANCH, prepare_timber
from src.utils.geometry import harvest_area


def list_datasets(folder):
    records, errors = [], []
    for path in sorted(Path(folder).glob('*/manifest.json')):
        try:
            record = json.loads(path.read_text())
            if record['id'] != path.parent.name or record['dataset_type'] != 'timber':
                raise ValueError('Invalid registry metadata')
            records.append(record)
        except (KeyError, ValueError, OSError) as exc:
            errors.append(f'{path.parent.name}: {exc}')
    return records, errors


def register_timber(raw, source_name, folder, recover=False):
    dataset = parse_hpr(raw, recover=recover)
    trees, logs, invalid = prepare_timber(dataset)
    identifier = hashlib.sha256(f'{dataset.fingerprint}:{BRANCH}:{recover}'.encode()).hexdigest()
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / identifier
    if (target / 'manifest.json').exists():
        return json.loads((target/'manifest.json').read_text())
    geojson = harvest_area(trees, dataset.fingerprint)
    manifest = {'id': identifier, 'name': Path(source_name).name, 'dataset_type': 'timber',
                'preparation_branch': BRANCH, 'source_sha256': dataset.fingerprint,
                'registered_at': datetime.now(timezone.utc).isoformat(), 'recovery_enabled': recover,
                'tree_count': len(trees), 'log_count': len(logs), 'area_ha': geojson['properties']['area_ha'],
                'schema': dataset.report['schema'], 'report': dataset.report, 'analytic_exclusions': invalid,
                'artifacts': ['source.hpr', 'trees.csv', 'timber_logs.csv', 'harvest_area.geojson']}
    # Files are visible to the picker only after the complete directory is committed.
    with tempfile.TemporaryDirectory(prefix='.preparing-', dir=folder) as staging:
        path = Path(staging)
        (path/'source.hpr').write_bytes(raw)
        trees.to_csv(path/'trees.csv', index=False)
        logs.to_csv(path/'timber_logs.csv', index=False)
        (path/'harvest_area.geojson').write_text(json.dumps(geojson, indent=2, allow_nan=False))
        (path/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False))
        try:
            os.rename(path, target)
        except OSError:
            if not (target/'manifest.json').exists():
                raise
    return manifest
