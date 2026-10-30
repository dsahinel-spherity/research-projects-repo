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
