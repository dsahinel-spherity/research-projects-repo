import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from src.preprocessing.timber import prepare_timber
from src.utils.geometry import harvest_area
from src.utils.traceability import search_logs, trace_id, point, selected_geojson, containment, download_name
from src.traceability_ui import trace_deck

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture
def timber(dataset):
    trees,logs,_ = prepare_timber(dataset)
    return trees,logs


def test_search_existing_unknown_and_missing(timber):
    _, logs = timber
    result = search_logs(logs,'LogscomID',' 105 ')
    assert len(result) == 1
    assert (result.iloc[0].StemKey,result.iloc[0].LogKey) == ('128373','1')
    assert search_logs(logs,'LogscomID','not-an-id').empty
    assert len(search_logs(logs,'LogscomID','')) == len(logs)
    assert len(search_logs(logs,'StemKey','128373')) >= 1
    missing = logs.loc[logs.LogscomID.isna()].iloc[0]
    assert not search_logs(logs,'StemKey',missing.StemKey).empty


def test_duplicate_ids_are_not_deduplicated(timber):
    _,logs = timber
    copied = logs.iloc[:2].copy()
    copied['LogscomID'] = 'duplicate'
    found = search_logs(copied,'LogscomID','duplicate')
    assert len(found)==2
    assert trace_id(found.iloc[0]) != trace_id(found.iloc[1])
    row = found.iloc[0].copy()
    row.DatasetID = 'another-source'
    assert trace_id(row) != trace_id(found.iloc[0])


def test_geojson_order_semantics_area_and_containment(timber):
    trees,logs = timber
    row = search_logs(logs,'LogscomID','105').iloc[0]
    area = harvest_area(trees,row.DatasetID)
    geo = selected_geojson(row,area)
    assert [f['geometry']['type'] for f in geo['features']] == ['Point','Point','LineString','Polygon']
    felling,product,line,polygon = geo['features']
    assert felling['geometry']['coordinates'] == [row.FellingLongitude,row.FellingLatitude]
    assert product['geometry']['coordinates'] == [row.DropOffLongitude,row.DropOffLatitude]
    assert felling['properties']['timestamp'] == row.FellingTime.isoformat()
    assert product['properties']['timestamp'] == row.DropOffTime.isoformat()
    assert 'not a confirmed drop-off' in product['properties']['interpretation']
    assert line['geometry']['coordinates'] == [point(row,'Felling'),point(row,'DropOff')]
    assert json.loads(json.dumps(geo,allow_nan=False))['type']=='FeatureCollection'
    assert download_name(row)=='log_105_traceability.geojson'
    assert polygon['properties']['area_ha']==pytest.approx(1.24405,abs=.0001)
    assert containment(point(row,'Felling'),area) in ['Inside','On boundary']
    assert containment(polygon['geometry']['coordinates'][0][0],area)=='On boundary'
    assert containment([0,0],area)=='Outside'


def test_missing_invalid_and_coincident_points(timber):
    _,logs = timber
    row=logs.iloc[0].copy()
    row.FellingLatitude=np.nan
    geo=selected_geojson(row)
    assert len(geo['features'])==1
    row.DropOffLongitude=181
    geo=selected_geojson(row)
    assert geo['features']==[]
    assert trace_deck(geo,{}) is None
    row.FellingLatitude=49
    row.FellingLongitude=7
    row.DropOffLatitude=49
    row.DropOffLongitude=7
    assert len(selected_geojson(row)['features'])==2
    assert containment(None,{'features':[]})=='Not assessable'


def test_deck_layers_and_synchronized_view(timber):
    trees,logs=timber
    row=search_logs(logs,'LogscomID','105').iloc[0]
    geo=selected_geojson(row,harvest_area(trees))
    visible={k:True for k in ['felling','product_position','straight_line_connection','estimated_harvest_area']}
    deck=trace_deck(geo,visible)
    parsed=json.loads(deck.to_json())
    assert len(parsed['layers'])==4
    dashes=next(l for l in parsed['layers'] if l['id']=='reference-connection')
    assert len(dashes['data'])==12
    assert dashes['widthUnits']=='pixels'
    assert dashes['widthMaxPixels']==2
    for layer in parsed['layers']:
        if layer['@@type']=='ScatterplotLayer':
            assert layer['radiusUnits']=='pixels'
            assert layer['radiusMaxPixels'] <= 10
    assert 7 < parsed['initialViewState']['longitude'] < 8
    visible['estimated_harvest_area']=False
    assert len(trace_deck(geo,visible).layers)==3
    visible['product_position']=False
    assert all(l.id!='product_position' for l in trace_deck(geo,visible).layers)


def test_search_page_user_workflow():
    app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=60).run()
    app.switch_page('pages/12_Log_Search.py').run()
    app.text_input(key='trace_query').set_value('105').run()
    assert not app.exception, [e.message for e in app.exception]
    assert app.metric[0].value=='105'
    assert app.checkbox(key='trace_layer_straight_line_connection_v2').value is False
    assert any('128373' in str(table.value) for table in app.dataframe)
    app.checkbox(key='trace_layer_estimated_harvest_area').uncheck().run()
    assert not app.exception
    app.text_input(key='trace_query').set_value('unknown').run()
    assert not app.exception
    assert any('No log matches' in m.value for m in app.info)
    assert len(app.metric)==0


def test_missing_coordinates_and_duplicate_search_ui(dataset,timber,monkeypatch):
    import src.ui as ui
    trees,logs=timber
    logs=logs.iloc[:2].copy()
    logs['LogscomID']='duplicate'
    logs[['FellingLatitude','DropOffLatitude']]=np.nan
    trees=trees.copy()
    trees['FellingLatitude']=np.nan
    monkeypatch.setattr(ui,'context',lambda:{'logs':logs,'trees':trees,'dataset':dataset,'signature':'missing-coords','volume':'Volume_price','volume_label':'Price volume (m³)'})
    app=AppTest.from_file(str(ROOT/'pages/12_Log_Search.py'),default_timeout=60).run()
    app.text_input(key='trace_query').set_value('duplicate').run()
    assert not app.exception
    assert any('2 logs match' in w.value for w in app.warning)
    assert len(app.metric)==0
    selection=next(s for s in app.selectbox if s.label.startswith('Matching log'))
    selection.set_value(trace_id(logs.iloc[0])).run()
    assert not app.exception, [e.message for e in app.exception]
    assert any('No valid coordinates' in m.value for m in app.info)
