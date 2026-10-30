import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest


def detect(frame, features, contamination=.05, seed=42):
    if not features or len(frame) < 10:
        raise ValueError('Select numeric features and at least ten logs.')
    if any(frame[c].notna().sum() == 0 for c in features):
        raise ValueError('Remove entirely missing features.')
    usable = frame[features].notna().any(axis=1)
    result = frame.loc[usable].copy()
    if len(result) < 10:
        raise ValueError('At least ten logs with observed measurements are required.')
    X = SimpleImputer(strategy='median').fit_transform(result[features])
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=seed)
    result['Anomaly'] = model.fit_predict(X) == -1
    result['AnomalyScore'] = -model.score_samples(X)
    med, std = result[features].median(), result[features].std().replace(0, np.nan)
    deviations = (result[features] - med).div(std)
    descriptions = []
    for _, row in deviations.iterrows():
        observed = row.dropna()
        strongest = observed.abs().nlargest(3).index
        descriptions.append('; '.join(f'{c}: {"above" if row[c] > 0 else "below"} median by {abs(row[c]):.2f} SD' for c in strongest) or 'Insufficient varying observed measurements')
    result['UnusualFeatures'] = descriptions
    result.attrs['excluded_all_missing'] = int((~usable).sum())
    return result.sort_values('AnomalyScore', ascending=False)
