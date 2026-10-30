from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GroupKFold, TimeSeriesSplit
from src.preprocessing.feature_engineering import CATEGORICAL_FEATURES, NUMERIC_FEATURES, PRECUT_CANDIDATES, frustum
from src.ml.evaluation import metrics, split_groups

ALGORITHMS = ['Linear Regression', 'Ridge Regression', 'Random Forest', 'HistGradientBoosting']
TARGETS = ['LogLength', 'Volume_price', 'Volume_ob', 'Volume_ub', 'CuttingDuration']

@dataclass
class Config:
    target: str
    features: list[str]
    algorithm: str = 'Random Forest'
    mode: str = 'Retrospective'
    confirmed_precut: bool = False
    seed: int = 42
    test_size: float = .25
    temporal: bool = False
    folds: int = 5
    n_estimators: int = 150
    max_depth: int = 8
    alpha: float = 1.
    learning_rate: float = .1


def eligible_features(target, mode='Retrospective'):
    if mode == 'Pre-cut':
        return PRECUT_CANDIDATES.copy()
    # Other recorded volume definitions are direct target proxies, never predictors.
    excluded = {target, 'Volume_price', 'Volume_ob', 'Volume_ub', 'CuttingDuration'}
    return [x for x in NUMERIC_FEATURES + CATEGORICAL_FEATURES if x not in excluded]


def make_pipeline(config):
    categorical = [x for x in config.features if x in CATEGORICAL_FEATURES]
    numeric = [x for x in config.features if x not in categorical]
    prep = ColumnTransformer([
        ('numeric', Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True)), ('scale', StandardScaler())]), numeric),
        ('category', Pipeline([('impute', SimpleImputer(strategy='constant', fill_value='Missing')), ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical),
    ])
    algorithms = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=config.alpha),
        'Random Forest': RandomForestRegressor(n_estimators=config.n_estimators, max_depth=config.max_depth, min_samples_leaf=2, random_state=config.seed, n_jobs=1),
        'HistGradientBoosting': HistGradientBoostingRegressor(max_depth=config.max_depth, learning_rate=config.learning_rate, random_state=config.seed),
    }
    return Pipeline([('preprocess', prep), ('model', algorithms[config.algorithm])])


def run_experiment(frame, config: Config):
    if config.target not in TARGETS or config.algorithm not in ALGORITHMS:
        raise ValueError('Unsupported target or algorithm.')
    if not config.features or not set(config.features) <= set(eligible_features(config.target, config.mode)):
        raise ValueError('Choose eligible features; target-derived and future-information features are prohibited.')
    if config.mode == 'Pre-cut' and not config.confirmed_precut:
        raise ValueError('The HPR schema does not establish pre-cut availability. Confirm DBH/species acquisition timing before training.')
    data = frame.dropna(subset=[config.target, 'TreeID']).reset_index(drop=True).copy()
    if len(data) < 15 or data[config.target].nunique() < 2:
        raise ValueError('Need at least 15 labeled logs with a varying target.')
    for col in config.features:
        if col in CATEGORICAL_FEATURES:
            data[col] = data[col].astype(object).where(data[col].notna(), np.nan)
        if data[col].notna().sum() == 0:
            raise ValueError(f'{col} has no observed values. Remove it from the feature set.')
    train, test = split_groups(data, config.test_size, config.seed, config.temporal)
    X, y = data[config.features], data[config.target]
    X_train, X_test = X.iloc[train], X.iloc[test]
    y_train, y_test = y.iloc[train], y.iloc[test]
    for col in config.features:
        if X_train[col].notna().sum() == 0:
            raise ValueError(f'{col} is entirely missing in the training partition.')
    if all(X_train[c].nunique() < 2 for c in config.features):
        raise ValueError('Selected training features have no variation; a meaningful predictive model is unsupported.')
    model = make_pipeline(config)
    groups = data.iloc[train].TreeID.reset_index(drop=True)
    k = min(config.folds, groups.nunique())
    if config.temporal:
        times = data.iloc[train].groupby('TreeID').HarvestDate.min()
        unique_times = sorted(times.unique())
        if len(unique_times) < 4:
            raise ValueError('At least four distinct training timestamps required for temporal cross-validation.')
        folds = []
        for a, b in TimeSeriesSplit(n_splits=min(k, len(unique_times)-1)).split(unique_times):
            ga = times[times.isin([unique_times[i] for i in a])].index
            gb = times[times.isin([unique_times[i] for i in b])].index
            folds.append((np.flatnonzero(groups.isin(ga)), np.flatnonzero(groups.isin(gb))))
    else:
        folds = list(GroupKFold(n_splits=k).split(X_train, y_train, groups))
    cv = []
    for a, b in folds:
        fitted = clone(model).fit(X_train.iloc[a], y_train.iloc[a])
        cv.append(metrics(y_train.iloc[b], fitted.predict(X_train.iloc[b])))
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    baseline = DummyRegressor(strategy='mean').fit(X_train, y_train)
    test_metrics = metrics(y_test, prediction)
    baseline_metrics = metrics(y_test, baseline.predict(X_test))
    meta = {'config': asdict(config), 'test': test_metrics, 'train': metrics(y_train, model.predict(X_train)),
            'baseline': baseline_metrics, 'cv': cv, 'labeled_rows': len(data), 'excluded_missing_targets': len(frame)-len(data),
            'train_rows': len(train), 'test_rows': len(test), 'train_trees': groups.nunique(),
            'test_trees': data.iloc[test].TreeID.nunique(), 'train_group_ids': sorted(groups.unique().tolist()),
            'test_group_ids': sorted(data.iloc[test].TreeID.unique().tolist())}
    if config.target.startswith('Volume_') and config.mode == 'Retrospective':
        bark = 'ub' if config.target == 'Volume_ub' else 'ob'
        geometric = frustum(data.iloc[test], bark)
        valid = geometric.notna()
        meta['geometric_baseline'] = {'bark': bark, 'rows': int(valid.sum()), 'metrics': metrics(y_test[valid], geometric[valid]) if valid.sum() > 1 else None}
    heldout = data.iloc[test].copy()
    heldout['Actual'], heldout['Predicted'] = y_test, prediction
    heldout['Residual'] = y_test - prediction
    return {'model': model, 'metadata': meta, 'X_train': X_train, 'X_test': X_test,
            'y_test': y_test, 'heldout': heldout, 'config': config}
