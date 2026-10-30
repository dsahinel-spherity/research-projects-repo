import numpy as np
import pandas as pd
from src.ingestion.stanford_parser import Dataset

NUMERIC_FEATURES = ['LogLength', 'ButtDiameter_ob', 'ButtDiameter_ub', 'MidDiameter_ob',
                    'MidDiameter_ub', 'TopDiameter_ob', 'TopDiameter_ub', 'DBH', 'ReferenceDiameter',
                    'StartPos', 'Volume_price', 'Volume_ob', 'Volume_ub', 'CuttingDuration']
CATEGORICAL_FEATURES = ['SpeciesGroupKey', 'ProductKey', 'StemGrade', 'CuttingReason', 'LogKey']
# These are only candidates: the UI requires explicit operational confirmation even for them.
PRECUT_CANDIDATES = ['DBH', 'SpeciesGroupKey']


def prepare(dataset: Dataset):
    if not dataset.report['analytics_safe']:
        raise ValueError('Duplicate or missing relational IDs: inspect and correct the source before analysis. Records are not silently deduplicated.')
    trees = dataset.trees.copy()
    logs = dataset.logs.copy()
    trees['FellCutDuration'] = (trees.FellCutEndTime - trees.FellCutStartTime).dt.total_seconds()
    logs['CuttingDuration'] = (logs.CuttingEndTime - logs.CuttingStartTime).dt.total_seconds()
    invalid = {}
    for frame, fields in [(trees, ['DBH', 'ReferenceDiameter', 'FellCutDuration']), (logs, NUMERIC_FEATURES)]:
        for col in fields:
            if col not in frame:
                continue
            bad = frame[col] < 0
            invalid[col] = int(bad.sum())
            frame.loc[bad, col] = np.nan
    for frame, prefixes in [(trees, ['Base', 'Crane']), (logs, ['Log'])]:
        for prefix in prefixes:
            for axis, limit in [('Latitude', 90), ('Longitude', 180)]:
                col = prefix + axis
                bad = frame[col].abs() > limit
                invalid[col] = int(bad.sum())
                frame.loc[bad, col] = np.nan
    joined = logs.merge(trees.drop(columns=['StemKey', 'MachineID']), on='TreeID', how='left', validate='many_to_one')
    return trees, joined, invalid


def frustum(frame, bark='ob'):
    """Length in cm; diameters in mm; volume in cubic meters. Missing stays missing."""
    length = frame['LogLength'] / 100
    butt, top = frame[f'ButtDiameter_{bark}'] / 1000, frame[f'TopDiameter_{bark}'] / 1000
    return np.pi * length / 12 * (butt**2 + butt * top + top**2)


def filter_data(trees, logs, selections=None, dates=None):
    selections = selections or {}
    t = trees.copy()
    if dates:
        start, end = dates
        t = t[t.HarvestDate.dt.date.between(start, end)]
    for col in ['SpeciesGroupKey', 'OperatorKey', 'HarvestObjectID']:
        if selections.get(col):
            t = t[t[col].isin(selections[col])]
    d = logs[logs.TreeID.isin(t.TreeID)].copy()
    for col in ['ProductKey', 'CuttingReason']:
        if selections.get(col):
            d = d[d[col].isin(selections[col])]
    if selections.get('ProductKey') or selections.get('CuttingReason'):
        t = t[t.TreeID.isin(d.TreeID)]
    return t, d
