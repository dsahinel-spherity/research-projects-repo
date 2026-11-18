"""Retrospective manual/automatic classification with tree-disjoint evaluation."""
from dataclasses import dataclass, asdict
import warnings
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from src.ml.evaluation import split_groups

LABELS = {'Automatic': 0, 'Other manual': 1}
FEATURES = ['LogLength', 'ButtDiameter_ob', 'MidDiameter_ob', 'TopDiameter_ob', 'DBH', 'ReferenceDiameter', 'SpeciesGroupKey', 'ProductKey']
CATEGORIES = ['SpeciesGroupKey', 'ProductKey']

@dataclass
class Config:
    features: list[str]
    C: float = 1.0
    balanced: bool = False
    seed: int = 42
    test_size: float = .25
    folds: int = 5
    algorithm: str = 'Logistic Regression'
    target: str = 'Manual vs automatic cuts'


def cut_labels(frame):
    # Never assume that every non-automatic reason proves a manual action.
    return frame.CuttingReason.astype('string').str.strip().map(LABELS)


def metrics(y, probability):
    prediction = probability >= .5
    both = len(np.unique(y)) == 2
    return {'Accuracy': float(accuracy_score(y, prediction)),
            'Balanced accuracy': float(balanced_accuracy_score(y, prediction)) if both else None,
            'Manual precision': float(precision_score(y, prediction, zero_division=0)),
            'Manual recall': float(recall_score(y, prediction, zero_division=0)),
            'Manual F1': float(f1_score(y, prediction, zero_division=0)),
            'ROC AUC': float(roc_auc_score(y, probability)) if both else None}


def run_experiment(frame, config):
    if not config.features or not set(config.features) <= set(FEATURES):
        raise ValueError('Choose supported measurement features; labels and identifiers cannot be predictors.')
    if not np.isfinite(config.C) or config.C <= 0 or not 2 <= config.folds <= 8:
        raise ValueError('C must be positive and CV folds must be between 2 and 8.')
    labels = cut_labels(frame)
    keep = labels.notna() & frame.TreeID.notna()
    data = frame.loc[keep].reset_index(drop=True).copy()
    y = labels.loc[keep].astype(int).reset_index(drop=True)
    if len(data) < 15 or y.nunique() != 2:
        raise ValueError('Need at least 15 labelled logs with both Automatic and Other manual cuts. Broaden the filters.')
    train, test = split_groups(data, config.test_size, config.seed)
    if y.iloc[train].nunique() != 2 or y.iloc[test].nunique() != 2:
        raise ValueError('Both classes must occur in training and held-out trees. Broaden the scope or choose a different seed.')
    X = data[config.features].copy()
    for col in config.features:
        if col in CATEGORIES:
            X[col] = X[col].astype(object).where(X[col].notna(), np.nan)
        if X.iloc[train][col].notna().sum() == 0:
            raise ValueError(f'{col} is entirely missing in training. Remove it.')
    if all(X.iloc[train][c].nunique() < 2 for c in config.features):
        raise ValueError('Training features have no variation.')
    cats = [c for c in config.features if c in CATEGORIES]
    nums = [c for c in config.features if c not in CATEGORIES]
    prep = ColumnTransformer([
        ('numeric', Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True)), ('scale', StandardScaler())]), nums),
        ('category', Pipeline([('impute', SimpleImputer(strategy='constant', fill_value='Missing')), ('encode', OneHotEncoder(handle_unknown='ignore'))]), cats),
    ])
    model = Pipeline([('preprocess', prep), ('model', LogisticRegression(C=config.C, class_weight='balanced' if config.balanced else None, max_iter=2000, random_state=config.seed))])
    groups = data.iloc[train].TreeID
    k = min(config.folds, groups.nunique(), int(y.iloc[train].value_counts().min()))
    if k < 2:
        raise ValueError('Too few minority-class training observations for cross-validation.')
    cv = []
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        try:
            for a, b in StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=config.seed).split(X.iloc[train], y.iloc[train], groups):
                if y.iloc[train].iloc[a].nunique() != 2:
                    raise ValueError('A CV training fold contains only one class. Broaden the scope or reduce folds.')
                if any(X.iloc[train].iloc[a][c].notna().sum() == 0 for c in config.features):
                    raise ValueError('A selected feature is entirely missing in a CV training fold.')
                fitted = clone(model).fit(X.iloc[train].iloc[a], y.iloc[train].iloc[a])
                cv.append(metrics(y.iloc[train].iloc[b], fitted.predict_proba(X.iloc[train].iloc[b])[:, 1]))
            model.fit(X.iloc[train], y.iloc[train])
        except ConvergenceWarning as exc:
            raise ValueError('Logistic regression did not converge. Reduce C or simplify the features.') from exc
    probability = model.predict_proba(X.iloc[test])[:, 1]
    baseline = DummyClassifier(strategy='prior').fit(X.iloc[train], y.iloc[train])
    heldout = data.iloc[test].copy()
    heldout['Actual'] = np.where(y.iloc[test] == 1, 'Manual', 'Automatic')
    heldout['Predicted'] = np.where(probability >= .5, 'Manual', 'Automatic')
    heldout['ManualProbability'] = probability
    metadata = {'task': 'binary_classification', 'config': asdict(config), 'label_mapping': LABELS,
                'mode': 'Retrospective', 'threshold': .5, 'class_counts': {'Automatic': int((y == 0).sum()), 'Manual': int((y == 1).sum())},
                'excluded_unmapped_labels': int(labels.isna().sum()), 'excluded_missing_tree_id': int((labels.notna() & frame.TreeID.isna()).sum()),
                'train_rows': len(train), 'test_rows': len(test),
                'train_group_ids': sorted(groups.unique().tolist()), 'test_group_ids': sorted(data.iloc[test].TreeID.unique().tolist()),
                'test': metrics(y.iloc[test], probability), 'train': metrics(y.iloc[train], model.predict_proba(X.iloc[train])[:, 1]),
                'baseline': metrics(y.iloc[test], baseline.predict_proba(X.iloc[test])[:, 1]), 'cv': cv,
                'confusion_matrix': confusion_matrix(y.iloc[test], probability >= .5, labels=[0, 1]).tolist()}
    return {'model': model, 'metadata': metadata, 'heldout': heldout}
