from pathlib import Path
from datetime import datetime, timezone
import json
import os
import uuid
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit


def metrics(actual, predicted):
    return {'MAE': float(mean_absolute_error(actual, predicted)),
            'RMSE': float(np.sqrt(mean_squared_error(actual, predicted))),
            'R2': float(r2_score(actual, predicted)) if len(actual) > 1 and np.var(actual) > 0 else None}


def split_groups(data, test_size=.25, seed=42, temporal=False):
    if data.TreeID.nunique() < 5:
        raise ValueError('At least five distinct trees are required for held-out evaluation and training-fold validation.')
    if temporal:
        if data.HarvestDate.isna().any():
            raise ValueError('Chronological validation requires a harvest timestamp for every tree.')
        times = data.groupby('TreeID').HarvestDate.agg(['min', 'max'])
        if not times['min'].equals(times['max']):
            raise ValueError('Inconsistent harvest timestamps within a tree.')
        ordered = times['min'].sort_values()
        cutoff = ordered.iloc[max(1, int(len(ordered) * (1 - test_size)))]
        test_groups = ordered[ordered >= cutoff].index
        test = np.flatnonzero(data.TreeID.isin(test_groups))
        train = np.flatnonzero(~data.TreeID.isin(test_groups))
    else:
        train, test = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed).split(data, groups=data.TreeID))
    if len(train) < 3 or len(test) < 2 or data.iloc[train].TreeID.nunique() < 3:
        raise ValueError('Insufficient training trees or held-out observations after splitting; broaden filters.')
    return train, test


def save_experiment(metadata, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    identifier = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:8]
    record = {**metadata, 'id': identifier, 'created_at': datetime.now(timezone.utc).isoformat()}
    target = folder / (identifier + '.json')
    temp = folder / (identifier + '.tmp')
    temp.write_text(json.dumps(record, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, target)
    return record


def history(folder):
    records, errors = [], []
    for path in sorted(Path(folder).glob('*.json'), reverse=True):
        try:
            records.append(json.loads(path.read_text()))
        except (ValueError, OSError) as exc:
            errors.append(f'{path.name}: {exc}')
    return records, errors
