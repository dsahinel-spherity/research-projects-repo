import json
import numpy as np
import pytest
from src.ml.feature_selection import Config, Selector, CLASS_TARGET, compare, applicable_methods

FEATURES = ['LogLength', 'MidDiameter_ob', 'DBH', 'Volume_price']

@pytest.mark.parametrize('target', ['CuttingDuration', CLASS_TARGET])
def test_methods_and_tree_isolation(data, target):
    r = compare(data, Config(target, FEATURES, applicable_methods(target), [1, 2], folds=3))
    assert not r['failures']
    assert set(r['train_group_ids']).isdisjoint(r['test_group_ids'])
    for split in r['cv_groups']:
        assert set(split['train']).isdisjoint(split['validation'])
        assert set(split['train'] + split['validation']) <= set(r['train_group_ids'])
    assert not r['recommendation'].startswith('PCA')
    for key, d in r['details'].items():
        assert len(d['cv']) == 3
        if key.startswith('PCA'):
            assert 0 < sum(d['pca_variance']) <= 1.000001
        else:
            assert set(d['selected']) <= set(FEATURES)
    json.dumps(r, allow_nan=False)


def test_holdout_does_not_choose_features(data):
    cfg = Config('CuttingDuration', FEATURES, ['F-test', 'Lasso'], [1, 2], folds=3)
    first = compare(data, cfg)
    changed = data.copy()
    changed.loc[changed.TreeID.isin(first['test_group_ids']), 'CuttingDuration'] += 1000
    second = compare(changed, cfg)
    assert first['comparison'] == second['comparison']
    assert first['recommendation'] == second['recommendation']
    key = first['recommendation']
    assert first['details'][key]['test'] != second['details'][key]['test']


def test_imputation_and_correlated_duplicates(data):
    X = data[['LogLength', 'DBH']].copy()
    X['Duplicate'] = X.LogLength
    X.loc[X.index[:8], 'DBH'] = np.nan
    cfg = Config('CuttingDuration', list(X), ['Correlation filter'], [3])
    selector = Selector('Correlation filter', 3, cfg).fit(X, data.CuttingDuration)
    assert not {'LogLength', 'Duplicate'} <= set(selector.selected)
    np.testing.assert_allclose(selector.imputer.statistics_, X.median())
    test = X.iloc[:2].copy(); test['DBH'] = np.nan
    assert np.isfinite(selector.transform(test)).all()


def test_invalid_scope_and_method(data):
    with pytest.raises(ValueError, match='eligible'):
        compare(data, Config('CuttingDuration', ['CuttingDuration'], ['F-test'], [1]))
    with pytest.raises(ValueError, match='appropriate'):
        compare(data, Config('CuttingDuration', FEATURES, ['Chi-square'], [1]))
    with pytest.raises(ValueError, match='both cut classes'):
        compare(data[data.CuttingReason == 'Automatic'], Config(CLASS_TARGET, FEATURES, ['F-test'], [1]))


def test_page_navigation_and_history(tmp_path, monkeypatch):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    import src.ml.evaluation as evaluation
    original = evaluation.save_experiment
    monkeypatch.setattr(evaluation, 'save_experiment', lambda m,f: original(m,tmp_path))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py'), default_timeout=90).run()
    app.toggle(key='advanced_tools').set_value(True).run()
    app.switch_page('pages/14_Feature_Selection.py').run()
    assert not app.exception
    next(w for w in app.multiselect if w.label=='Methods to compare').set_value(['F-test'])
    next(w for w in app.multiselect if w.label=='Subset sizes / PCA component counts to try').set_value([1,2])
    next(b for b in app.button if b.label=='Compare feature sets').click().run()
    assert not app.exception, [e.message for e in app.exception]
    assert app.success
    assert 'feature_selection_result' in app.session_state
    assert len(list(tmp_path.glob('*.json'))) == 1
    app.switch_page('pages/9_Experiment_History.py').run()
    assert not app.exception
    app.switch_page('pages/14_Feature_Selection.py').run()
    assert 'feature_selection_result' in app.session_state
    app.multiselect(key='filter_CuttingReason').set_value(['Automatic']).run()
    assert 'feature_selection_result' not in app.session_state


def test_regression_evaluation_ignores_other_experiment_types(monkeypatch):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    import src.ml.evaluation as evaluation
    import src.ui as ui
    original = ui.context
    state = {}
    def capture():
        c = original(); state.update(c); return c
    monkeypatch.setattr(ui, 'context', capture)
    monkeypatch.setattr(evaluation, 'history', lambda folder: ([
        {'scope_signature': state['signature'], 'config': {'algorithm': name}}
        for name in ['Feature Selection', 'Logistic Regression']], []))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/'pages/8_Model_Evaluation.py'), default_timeout=60).run()
    assert not app.exception
    assert any('Train models' in x.value for x in app.info)
