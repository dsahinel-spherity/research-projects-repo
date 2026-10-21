"""Synthetic-only experiments with held-out evaluation and train-only preprocessing."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, f1_score, mean_absolute_error, r2_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.services.data_service import load_mock_journeys

QUESTIONS = {
    'Which journeys are more likely to open a document?': ('opened_document', 'classifier', 'Compare navigation patterns associated with document opening. Test clearer document links in a real UX experiment.'),
    'Which journeys meet the deep-engagement definition?': ('deep_engagement', 'classifier', 'Estimate the synthetic engagement label from entry context. This label is defined by sections, documents and duration.'),
    'Which entry contexts are associated with longer journeys?': ('duration_seconds', 'regressor', 'Explore expected journey duration from entry context. Longer is not automatically better: it may reflect interest or friction.'),
    'Which entry contexts are associated with more section exploration?': ('section_count', 'regressor', 'Explore navigation depth from entry context. Validate useful exploration against successful document discovery.'),
    'Which journeys deserve a closer look?': (None, 'anomaly', 'Prioritize unusual patterns for review. An unusual journey is not proof of a bot, frustration or an error.'),
}
ENTRY_FEATURES = ['source', 'device', 'country', 'first_section']


def preparation_report(raw, data, X, fitted, target, fit_rows, evaluation_rows):
    """Describe the actual fitted preprocessing without exposing model objects."""
    replacements = []
    for name, transformer, columns in fitted.transformers_:
        if name not in {'numeric', 'category'} or not columns:
            continue
        values = transformer.named_steps['impute'].statistics_
        for column, value in zip(columns, values):
            replacements.append({
                'Input': column,
                'Missing values filled': int(X[column].isna().sum()),
                'Method': 'Median' if name == 'numeric' else 'Most frequent value',
                'Replacement learned from fitting data': str(value),
            })
    unique_rows = len(raw.drop_duplicates())
    return {
        'original_rows': len(raw),
        'duplicates_removed': len(raw) - unique_rows,
        'missing_target_rows_removed': unique_rows - len(data),
        'retained_rows': len(data),
        'target': target,
        'inputs': X.columns.tolist(),
        'excluded_inputs': [c for c in raw.columns if c not in X.columns and c != target],
        'missing_values_filled': int(X.isna().sum().sum()),
        'replacements': replacements,
        'encoded_columns': len(fitted.get_feature_names_out()),
        'fit_rows': fit_rows,
        'evaluation_rows': evaluation_rows,
    }


def run_experiment(question, model_name='Random forest', test_size=0.25, contamination=0.05):
    if question not in QUESTIONS:
        raise ValueError('Choose a supported synthetic experiment.')
    target, kind, action = QUESTIONS[question]
    raw = load_mock_journeys()
    data = raw.drop_duplicates().reset_index(drop=True)
    features = ENTRY_FEATURES.copy()
    if target == 'opened_document':
        features += ['section_count', 'unique_section_count']
    if kind == 'anomaly':
        features += ['section_count', 'unique_section_count', 'duration_seconds', 'document_open_count']
    # Outcome-derived fields never enter supervised entry-context models.
    if target:
        data = data.dropna(subset=[target]).reset_index(drop=True)
    X = data[features].copy()
    # Treat Python None and pandas missing values consistently in sklearn.
    X = X.where(X.notna(), np.nan)
    numeric = X.select_dtypes(include='number').columns.tolist()
    categorical = [c for c in features if c not in numeric]
    preprocess = ColumnTransformer([
        ('numeric', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), numeric),
        ('category', Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical),
    ])
    if kind == 'anomaly':
        pipeline = Pipeline([('prepare', preprocess), ('model', IsolationForest(n_estimators=200, contamination=contamination, random_state=42, n_jobs=1))])
        labels = pipeline.fit_predict(X)
        result = data.copy()
        result['review_score'] = -pipeline.decision_function(X)
        result['flagged'] = labels == -1
        preparation = preparation_report(raw, data, X, pipeline.named_steps['prepare'], target, len(X), 0)
        return dict(kind=kind, action=action, preparation=preparation, rows=len(data), features=features, records=result.sort_values('review_score', ascending=False), flagged=int((labels == -1).sum()))
    y = data[target]
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y if kind == 'classifier' else None)
    if kind == 'classifier':
        estimator = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, class_weight='balanced', random_state=42, n_jobs=1) if model_name == 'Random forest' else LogisticRegression(max_iter=1500, class_weight='balanced', random_state=42)
        baseline = DummyClassifier(strategy='most_frequent')
    else:
        estimator = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=42, n_jobs=1) if model_name == 'Random forest' else Ridge(alpha=10)
        baseline = DummyRegressor(strategy='median')
    pipeline = Pipeline([('prepare', preprocess), ('model', estimator)])
    pipeline.fit(train_x, train_y)
    preparation = preparation_report(raw, data, X, pipeline.named_steps['prepare'], target, len(train_x), len(test_x))
    prediction = pipeline.predict(test_x)
    baseline.fit(np.zeros((len(train_y), 1)), train_y)
    reference = baseline.predict(np.zeros((len(test_y), 1)))
    if kind == 'classifier':
        metrics = {'Balanced accuracy': balanced_accuracy_score(test_y, prediction), 'Baseline balanced accuracy': balanced_accuracy_score(test_y, reference), 'F1 score': f1_score(test_y, prediction, zero_division=0)}
        matrix = confusion_matrix(test_y, prediction, labels=[0, 1])
        confusion = pd.DataFrame([{'Actual': a, 'Predicted': p, 'Journeys': int(matrix[i, j])} for i, a in enumerate(['No', 'Yes']) for j, p in enumerate(['No', 'Yes'])])
        scoring = 'balanced_accuracy'
    else:
        metrics = {'Mean absolute error': mean_absolute_error(test_y, prediction), 'Baseline error': mean_absolute_error(test_y, reference), 'R²': r2_score(test_y, prediction)}
        confusion = pd.DataFrame()
        scoring = 'neg_mean_absolute_error'
    importance = permutation_importance(pipeline, test_x, test_y, n_repeats=5, random_state=42, scoring=scoring, n_jobs=1)
    return dict(kind=kind, target=target, action=action, preparation=preparation, train_rows=len(train_y), test_rows=len(test_y), features=features, metrics=metrics, confusion=confusion,
                importance=pd.DataFrame({'Feature': features, 'Importance': importance.importances_mean, 'Variation': importance.importances_std}),
                predictions=pd.DataFrame({'Actual': test_y, 'Predicted': prediction}).reset_index(drop=True))
