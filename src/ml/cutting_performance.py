"""Retrospective cutting benchmarks with tree-separated fitting/calibration/evaluation."""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from src.ml.regression import Config, make_pipeline
from src.ml.evaluation import metrics

FEATURES = ['LogLength', 'ButtDiameter_ob', 'TopDiameter_ob', 'DBH']


def evaluate_cutting(frame, algorithm='Random Forest', coverage=.9):
    if algorithm not in ['Random Forest', 'Ridge Regression'] or not .5 < coverage < 1:
        raise ValueError('Choose Random Forest or Ridge and coverage between 0.5 and 1.')
    d = frame.copy().reset_index(drop=True)
    d['SourceRow'] = np.arange(len(d))
    for f in FEATURES + ['CuttingDuration']:
        d[f] = pd.to_numeric(d[f], errors='coerce').replace([np.inf, -np.inf], np.nan)
    valid = d.TreeID.notna() & d.CuttingDuration.gt(0) & d[FEATURES].notna().any(axis=1) & ~(d[FEATURES] < 0).any(axis=1)
    excluded = d.loc[~valid].copy()
    excluded['ExclusionReason'] = ['; '.join(reason for condition, reason in [(pd.isna(row.TreeID), 'Missing tree identity'), (pd.isna(row.CuttingDuration) or row.CuttingDuration <= 0, 'Missing or nonpositive duration'), (row[FEATURES].isna().all(), 'All inputs missing'), ((row[FEATURES] < 0).any(), 'Negative input')] if condition) for _,row in excluded.iterrows()]
    d = d.loc[valid].reset_index(drop=True)
    if len(d) < 30 or d.TreeID.nunique() < 10 or d.CuttingDuration.nunique() < 2:
        raise ValueError('Need 30 eligible logs, 10 distinct trees and varying positive cutting durations. Broaden filters.')
    for name in ['ExpectedSeconds', 'LowerSeconds', 'UpperSeconds', 'BaselineSeconds', 'Fold']:
        d[name] = np.nan
    audit = []
    for fold, (development, test) in enumerate(GroupKFold(n_splits=5).split(d, groups=d.TreeID), 1):
        dev = d.iloc[development]
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=42).split(dev, groups=dev.TreeID))
        train, calibration = development[a], development[b]
        available = [f for f in FEATURES if d.iloc[train][f].notna().any()]
        if not available or all(d.iloc[train][f].nunique() < 2 for f in available):
            raise ValueError('Training measurements have insufficient variation for this scope.')
        config = Config('CuttingDuration', available, algorithm=algorithm)
        model = make_pipeline(config).fit(d.iloc[train][available], d.iloc[train].CuttingDuration)
        cal_prediction = np.maximum(0, model.predict(d.iloc[calibration][available]))
        residual = np.abs(d.iloc[calibration].CuttingDuration.to_numpy()-cal_prediction)
        # Separate calibration trees; empirical log-level range, not a group-coverage guarantee.
        radius = float(np.quantile(residual, coverage, method='higher'))
        pred = np.maximum(0, model.predict(d.iloc[test][available]))
        d.loc[test, 'ExpectedSeconds'] = pred
        d.loc[test, 'LowerSeconds'] = np.maximum(0, pred-radius)
        d.loc[test, 'UpperSeconds'] = pred+radius
        d.loc[test, 'BaselineSeconds'] = float(d.iloc[train].CuttingDuration.median())
        d.loc[test, 'Fold'] = fold
        audit.append({'fold': fold, 'fit_trees': sorted(d.iloc[train].TreeID.unique().tolist()),
                      'calibration_trees': sorted(d.iloc[calibration].TreeID.unique().tolist()),
                      'evaluation_trees': sorted(d.iloc[test].TreeID.unique().tolist()),
                      'features': available, 'radius_seconds': radius})
    d['DifferenceSeconds'] = d.CuttingDuration-d.ExpectedSeconds
    d['Review'] = d.CuttingDuration > d.UpperSeconds
    d['ImputedInputCount'] = d[FEATURES].isna().sum(axis=1)
    measured_coverage = float(((d.CuttingDuration >= d.LowerSeconds) & (d.CuttingDuration <= d.UpperSeconds)).mean())
    summary = {'algorithm': algorithm, 'features': FEATURES, 'rows': len(d), 'trees': int(d.TreeID.nunique()),
               'excluded_rows': len(excluded), 'model': metrics(d.CuttingDuration, d.ExpectedSeconds),
               'median_baseline': metrics(d.CuttingDuration, d.BaselineSeconds), 'requested_coverage': coverage,
               'observed_coverage': measured_coverage, 'review_logs': int(d.Review.sum()), 'folds': audit}
    return {'records': d.sort_values('DifferenceSeconds', ascending=False), 'excluded': excluded, 'summary': summary}
