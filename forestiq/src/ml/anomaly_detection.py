"""Exploratory Isolation Forest screening of timber measurements."""
import math
import warnings
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning


def detect(frame, features, contamination=.05, seed=42, *, method="Isolation Forest", bottleneck=1):
    if len(features) < 2 or len(set(features)) != len(features):
        raise ValueError('Select at least two distinct measurements.')
    if not set(features) <= set(frame.columns):
        raise ValueError('Selected measurements are missing from this dataset.')
    if not 0 < contamination <= .5:
        raise ValueError('Contamination must be between 0 and 0.5.')
    raw = frame[list(features)].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    usable = raw.notna().any(axis=1) & ~(raw < 0).any(axis=1)
    if usable.sum() < 10:
        raise ValueError('At least 10 usable logs are needed.')
    if (raw.loc[usable].nunique() < 2).any():
        raise ValueError('Remove empty or constant measurements.')
    values = SimpleImputer(strategy='median').fit_transform(raw.loc[usable])
    z = StandardScaler().fit_transform(values)
    if method == 'Isolation Forest':
        model = IsolationForest(n_estimators=200, contamination=contamination, random_state=seed)
        flags = model.fit_predict(z) == -1
        scores = -model.score_samples(z)
        contributions = np.abs(z)
        explanation = 'Largest standardized departure (descriptive, not model attribution)'
    elif method == 'Autoencoder':
        if not 1 <= bottleneck < len(features):
            raise ValueError('Autoencoder bottleneck must be smaller than the number of measurements.')
        model = MLPRegressor(hidden_layer_sizes=(bottleneck,), activation='tanh', solver='lbfgs',
                             alpha=.01, max_iter=2000, max_fun=50000, random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            try:
                model.fit(z, z)
            except ConvergenceWarning as exc:
                raise ValueError('Autoencoder did not converge. Try fewer measurements or another bottleneck size.') from exc
        contributions = (z - model.predict(z))**2
        scores = contributions.mean(axis=1)
        flags = np.zeros(len(scores), dtype=bool)
        flags[np.argsort(-scores, kind='stable')[:math.ceil(contamination * len(scores))]] = True
        explanation = 'Largest observed reconstruction error'
    else:
        raise ValueError('Unknown anomaly detection method.')
    out = frame.copy()
    out['Anomaly'] = False
    out['AnomalyScore'] = np.nan
    out['PC1'], out['PC2'] = np.nan, np.nan
    out['UnusualFeatures'] = pd.Series(None, index=out.index, dtype='object')
    out['ScoringStatus'] = np.where(~usable, 'Unscorable', np.where(raw.isna().any(axis=1), 'Median-imputed', 'Scored'))
    out.loc[usable, 'Anomaly'] = flags
    out.loc[usable, 'AnomalyScore'] = scores
    out.loc[usable, ['PC1', 'PC2']] = PCA(n_components=2).fit_transform(z)
    departures = np.where(raw.loc[usable].notna(), contributions, -np.inf)
    out.loc[usable, 'UnusualFeatures'] = [features[i] for i in departures.argmax(axis=1)]
    out['MissingMeasurements'] = raw.isna().sum(axis=1)
    out.attrs.update(method=method, features=list(features), fraction=contamination, explanation=explanation)
    return out.sort_values('AnomalyScore', ascending=False, na_position='last')


def detect_autoencoder(frame, features, contamination=.05, seed=42, bottleneck=1):
    """Fit a bottleneck neural network on real logs and rank reconstruction errors."""
    return detect(frame, features, contamination, seed, method='Autoencoder', bottleneck=bottleneck)
