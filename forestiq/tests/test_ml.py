import json
from src.ml.classification import Config as ClassificationConfig, cut_labels, run_experiment as run_classification
import numpy as np
import pytest
from src.ml.regression import Config, ALGORITHMS, run_experiment
from src.ml.evaluation import split_groups, save_experiment, history
from src.ml.anomaly_detection import detect
from src.ml.clustering import segment
from src.explainability.shap_engine import explain, permutation

@pytest.mark.parametrize('algorithm', ALGORITHMS)
def test_regression_group_integrity(data, algorithm):
    r = run_experiment(data, Config('LogLength', ['DBH', 'SpeciesGroupKey', 'ButtDiameter_ob'], algorithm=algorithm, n_estimators=20, folds=3))
    m = r['metadata']
    assert set(m['train_group_ids']).isdisjoint(m['test_group_ids'])
    assert np.isfinite(m['test']['MAE'])
    assert len(m['cv']) == 3
    assert len(r['heldout']) == m['test_rows']


def test_temporal_split(data):
    a,b = split_groups(data, temporal=True)
    assert data.iloc[a].HarvestDate.max() < data.iloc[b].HarvestDate.min()
    assert set(data.iloc[a].TreeID).isdisjoint(data.iloc[b].TreeID)
    r = run_experiment(data, Config('CuttingDuration', ['LogLength', 'DBH'], algorithm='Ridge Regression', temporal=True, folds=3))
    assert len(r['metadata']['cv']) == 3


def test_leakage_policy(data):
    with pytest.raises(ValueError, match='Confirm'):
        run_experiment(data, Config('LogLength', ['DBH'], mode='Pre-cut'))
    with pytest.raises(ValueError, match='eligible'):
        run_experiment(data, Config('LogLength', ['ProductKey'], mode='Pre-cut', confirmed_precut=True))
    with pytest.raises(ValueError, match='eligible'):
        run_experiment(data, Config('Volume_price', ['Volume_ob']))
    with pytest.raises(ValueError, match='eligible'):
        run_experiment(data, Config('LogLength', ['LogLength']))


def test_volume_baseline_persistence_and_explanations(data, tmp_path):
    r = run_experiment(data, Config('Volume_price', ['LogLength', 'ButtDiameter_ob', 'TopDiameter_ob'], algorithm='Ridge Regression', folds=3))
    assert r['metadata']['geometric_baseline']['rows'] == len(r['heldout'])
    saved = save_experiment(r['metadata'], tmp_path)
    records, errors = history(tmp_path)
    assert len(records) == 1 and not errors and records[0]['id'] == saved['id']
    assert len(permutation(r)) == 3
    values, positions = explain(r, max_rows=3)
    predictions = r['model'].predict(r['X_test'].iloc[positions])
    assert np.allclose(values.base_values + values.values.sum(axis=1), predictions)


def test_anomalies_and_clusters(data):
    features = ['LogLength', 'MidDiameter_ob', 'Volume_price']
    d = detect(data, features)
    assert 0 < d.Anomaly.sum() < len(d)
    assert d.AnomalyScore.is_monotonic_decreasing
    for algorithm in ['K-Means', 'DBSCAN']:
        out, profiles, score, descriptions, variance = segment(data, features, algorithm)
        assert len(out) == len(data)
        assert set(['PC1', 'PC2', 'Cluster']) <= set(out)
        assert len(descriptions) == len(profiles)


def test_tiny_data_rejected(data):
    with pytest.raises(ValueError):
        run_experiment(data.head(5), Config('LogLength', ['DBH']))



CLASSIFICATION_FEATURES = ['LogLength', 'MidDiameter_ob', 'DBH']

def test_classifier_real_sample(data):
    result = run_classification(data, ClassificationConfig(CLASSIFICATION_FEATURES))
    m = result['metadata']
    assert m['class_counts'] == {'Automatic': 245, 'Manual': 78}
    assert not set(m['train_group_ids']) & set(m['test_group_ids'])
    assert result['heldout'].ManualProbability.between(0, 1).all()
    assert np.asarray(m['confusion_matrix']).sum() == m['test_rows']
    assert len(m['cv']) == 5
    # Same rows, seed and fraction must reproduce the regression split exactly.
    train_ids, test_ids = split_groups(data, test_size=.25, seed=42)
    assert set(m['train_group_ids']) == set(data.iloc[train_ids].TreeID)
    assert set(m['test_group_ids']) == set(data.iloc[test_ids].TreeID)
    for metric in ['Manual precision', 'Manual recall', 'Balanced accuracy', 'ROC AUC']:
        assert 0 <= m['test'][metric] <= 1
    json.dumps(m, allow_nan=False)
    # Imputation uses training values only, never held-out measurements.
    train = data[data.TreeID.isin(m['train_group_ids'])]
    np.testing.assert_allclose(result['model']['preprocess'].named_transformers_['numeric']['impute'].statistics_, train[CLASSIFICATION_FEATURES].median())


def test_unknown_labels_and_leakage(data):
    sample = data.copy()
    sample.loc[sample.index[:3], 'CuttingReason'] = ['Unknown', None, 'Decay']
    assert cut_labels(sample).iloc[:3].isna().all()
    with pytest.raises(ValueError, match='predictors'):
        run_classification(data, ClassificationConfig(['CuttingReason']))
    with pytest.raises(ValueError, match='both Automatic'):
        run_classification(data[data.CuttingReason == 'Automatic'], ClassificationConfig(CLASSIFICATION_FEATURES))


def test_missing_and_categorical_predictors(data):
    sample = data.copy()
    sample.loc[sample.index[::3], 'DBH'] = np.nan
    sample.loc[sample.index[::4], 'ProductKey'] = None
    result = run_classification(sample, ClassificationConfig(['DBH', 'LogLength', 'ProductKey']))
    assert result['heldout'].ManualProbability.notna().all()
    sample['DBH'] = np.nan
    with pytest.raises(ValueError, match='entirely missing'):
        run_classification(sample, ClassificationConfig(['DBH']))


def test_classifier_navigation_and_invalidation(tmp_path, monkeypatch):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    import src.ml.evaluation as evaluation
    original = evaluation.save_experiment
    monkeypatch.setattr(evaluation, 'save_experiment', lambda meta, folder: original(meta, tmp_path))
    root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(str(root/'app.py'), default_timeout=60).run()
    app.toggle(key='advanced_tools').set_value(True).run()
    app.switch_page('pages/13_Cut_Classification.py').run()
    assert app.title[0].value == 'Logistic Regression'
    next(b for b in app.button if b.label == 'Train cut classifier').click().run()
    assert not app.exception, [e.message for e in app.exception]
    assert len(app.success) == 1
    assert 'cut_classifier_result' in app.session_state
    assert len(list(tmp_path.glob('*.json'))) == 1
    app.switch_page('pages/9_Experiment_History.py').run()
    assert not app.exception
    app.switch_page('pages/13_Cut_Classification.py').run()
    assert 'cut_classifier_result' in app.session_state
    app.multiselect(key='filter_CuttingReason').set_value(['Automatic']).run()
    assert 'cut_classifier_result' not in app.session_state
    next(b for b in app.button if b.label == 'Train cut classifier').click().run()
    assert not app.exception
    assert app.error


def test_autoencoder_real_logs(data):
    import math
    from src.ml.anomaly_detection import detect_autoencoder
    features = ['LogLength', 'MidDiameter_ob', 'Volume_price']
    result = detect_autoencoder(data, features)
    repeated = detect_autoencoder(data, features)
    assert len(result) == len(data)
    assert np.isfinite(result.AnomalyScore).all()
    assert result.AnomalyScore.ge(0).all()
    assert result.Anomaly.sum() == math.ceil(.05 * len(data))
    np.testing.assert_allclose(result.AnomalyScore, repeated.AnomalyScore)
    assert set(result.UnusualFeatures) <= set(features)
    assert result.attrs['method'] == 'Autoencoder'
    with pytest.raises(ValueError, match='bottleneck'):
        detect_autoencoder(data, features, bottleneck=3)
