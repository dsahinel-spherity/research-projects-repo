from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest
ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('page', sorted((ROOT / 'pages').glob('*.py')))
def test_page_smoke(page):
    app = AppTest.from_file(str(page), default_timeout=60).run()
    assert not app.exception, '\n'.join(e.message for e in app.exception)


def test_navigation():
    app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=60).run()
    assert not app.exception
    assert len(app.metric) == 4


def test_anomaly_interaction():
    app = AppTest.from_file(str(ROOT / 'pages/5_Anomaly_Detection.py'), default_timeout=60).run()
    app.button[0].click().run()
    assert not app.exception
    assert len(app.metric) == 3


def test_cluster_interaction():
    app = AppTest.from_file(str(ROOT / 'pages/6_Clustering.py'), default_timeout=60).run()
    app.button[0].click().run()
    assert not app.exception
    assert len(app.metric) == 1


def test_training_explanation_navigation(tmp_path, monkeypatch):
    import src.ml.evaluation as evaluation
    original = evaluation.save_experiment
    monkeypatch.setattr(evaluation, 'save_experiment', lambda meta, folder: original(meta, tmp_path))
    app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=90).run()
    app.toggle(key='advanced_tools').set_value(True).run()
    app.switch_page('pages/3_Predictive_Analytics.py').run()
    app.button[0].click().run(timeout=90)
    assert not app.exception
    assert len(app.success) == 1
    assert 'active_experiment' in app.session_state
    app.switch_page('pages/4_Explainable_AI.py').run(timeout=90)
    assert not app.exception
    app.button[0].click().run(timeout=90)
    assert not app.exception
    assert 'shap_result' in app.session_state
    app.switch_page('pages/9_Experiment_History.py').run()
    assert not app.exception
    app.switch_page('pages/1_Overview.py').run()
    app.multiselect(key='filter_CuttingReason').set_value(['Other manual']).run()
    assert not app.exception
    assert 'active_experiment' not in app.session_state


def test_timber_page_rebuilds_pre_timber_cache(dataset, monkeypatch):
    """A long-lived server must reparse old cached tables, not hide missing columns."""
    import copy
    from unittest.mock import Mock
    import src.ui as ui
    stale = copy.deepcopy(dataset)
    stale.logs = stale.logs.drop(columns=['ProductName', 'LogCRS', 'LogReceiverPosition', 'LogCoordinateDate'])
    stale.trees = stale.trees.drop(columns=['ObjectKey', 'CraneCRS', 'CraneReceiverPosition', 'CraneCoordinateDate'])
    stale.report.pop('schema')
    cache = Mock(return_value=stale)
    monkeypatch.setattr(ui, '_cached_parse', cache)
    app = AppTest.from_file(str(ROOT / 'pages/10_Timber_Data.py'), default_timeout=60).run()
    assert not app.exception, '\n'.join(e.message for e in app.exception)
    cache.clear.assert_called_once()
    table = app.dataframe[0].value
    assert len(table) == 323
    assert 'ProductName' in table
    assert table.FellingLatitude.notna().all()
    assert table.DropOffTime.notna().all()


def test_beginner_navigation_to_timber():
    app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=60).run()
    app.switch_page('pages/10_Timber_Data.py').run()
    assert not app.exception
    assert len(app.dataframe[0].value) == 323
    app.switch_page('pages/11_Analytics.py').run()
    assert not app.exception


def test_dataset_and_filters_persist_between_sections(tmp_path, monkeypatch):
    import src.ui as ui
    from src.datasets.registry import register_timber
    from tests.test_timber import complete_hpr_fixture
    # A different, complete file ensures we verify actual loaded rows, not only labels.
    record = register_timber(complete_hpr_fixture(), 'second-dataset.hpr', tmp_path/'data/registry')
    monkeypatch.setattr(ui, 'ROOT', tmp_path)
    app = AppTest.from_file(str(ROOT/'app.py'), default_timeout=60).run()
    app.selectbox(key='source_dataset').select(record['id']).run()
    assert not app.exception
    assert app.metric[0].value == '1'
    app.multiselect(key='filter_ProductKey').set_value(['10353']).run()
    volume = next(x for x in app.selectbox if x.label=='Recorded volume definition')
    volume.select('Under-bark volume (m³)').run()
    for page in ['pages/10_Timber_Data.py','pages/12_Log_Search.py','pages/11_Analytics.py','pages/1_Overview.py']:
        app.switch_page(page).run()
        assert not app.exception, [e.message for e in app.exception]
        assert app.selectbox(key='source_dataset').value == record['id']
        assert app.multiselect(key='filter_ProductKey').value == ['10353']
        assert next(x for x in app.selectbox if x.label=='Recorded volume definition').value == 'Under-bark volume (m³)'
    assert app.metric[0].value == '1'


def test_uploaded_source_survives_navigation_and_can_be_cleared():
    from tests.test_timber import complete_hpr_fixture
    app = AppTest.from_file(str(ROOT/'app.py'), default_timeout=60)
    # AppTest cannot upload files; seed the bytes retained by the uploader callback.
    app.session_state['_uploaded_source'] = {'raw':complete_hpr_fixture(), 'name':'active-upload.hpr'}
    app.run()
    assert not app.exception
    assert app.metric[0].value == '1'
    for page in ['pages/10_Timber_Data.py', 'pages/11_Analytics.py', 'pages/1_Overview.py']:
        app.switch_page(page).run()
        assert not app.exception
        assert app.session_state['_uploaded_source']['name'] == 'active-upload.hpr'
        assert any('active-upload.hpr' in text.value for text in app.caption)
    assert app.metric[0].value == '1'
    next(b for b in app.button if b.label=='Clear uploaded file').click().run()
    assert not app.exception
    assert '_uploaded_source' not in app.session_state
    assert app.metric[0].value == '56'
