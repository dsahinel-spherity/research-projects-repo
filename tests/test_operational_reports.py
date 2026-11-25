import numpy as np
import pandas as pd
import pytest
from src.ml.cutting_performance import evaluate_cutting
from src.datasets.quality_report import build_quality_report


def test_cutting_group_separation_and_predictions(data):
    result=evaluate_cutting(data)
    d=result['records'];s=result['summary']
    assert len(d)==int(data.CuttingDuration.gt(0).sum())
    assert d.SourceRow.nunique()==len(d)
    assert d.ExpectedSeconds.notna().all()
    assert (d.LowerSeconds<=d.ExpectedSeconds).all()
    assert (d.UpperSeconds>=d.ExpectedSeconds).all()
    assert d.Review.equals(d.CuttingDuration>d.UpperSeconds)
    evaluation=[]
    for f in s['folds']:
        fit,cal,test=map(set,[f['fit_trees'],f['calibration_trees'],f['evaluation_trees']])
        assert not fit & cal and not fit & test and not cal & test
        evaluation+=f['evaluation_trees']
    assert len(evaluation)==len(set(evaluation))==data.loc[data.CuttingDuration.gt(0)].TreeID.nunique()
    # Evaluation targets cannot influence their own predictions or calibrated range.
    ids=s['folds'][0]['evaluation_trees']
    altered=data.copy()
    altered.loc[altered.TreeID.isin(ids),'CuttingDuration']*=2
    changed=evaluate_cutting(altered)['records'].set_index('SourceRow')
    original=d.set_index('SourceRow')
    positions=original.index[original.TreeID.isin(ids)]
    np.testing.assert_allclose(original.loc[positions,['ExpectedSeconds','LowerSeconds','UpperSeconds']],changed.loc[positions,['ExpectedSeconds','LowerSeconds','UpperSeconds']])


def test_cutting_guards(data):
    with pytest.raises(ValueError):evaluate_cutting(data.head(10))
    with pytest.raises(ValueError):evaluate_cutting(data,coverage=1)
    d=data.copy();d.loc[d.index[0],'CuttingDuration']=-1
    assert evaluate_cutting(d,'Ridge Regression')['summary']['excluded_rows']==int(data.CuttingDuration.le(0).sum())+1


def test_quality_real_and_unavailable(data):
    r=build_quality_report(data)
    assert 'volumes' not in r
    assert 'Volume inconsistency' not in r['issues'].Check.tolist()
    d=data.copy();d.loc[d.index[0],'Volume_ob']=np.nan
    changed=build_quality_report(d)
    assert 'Missing/nonfinite Volume_ob' in changed['issues'].Check.tolist()
    d.loc[d.index[1],'LogLength']=-1
    assert 'Negative LogLength' in build_quality_report(d)['issues'].Check.tolist()
    empty=build_quality_report(data.head(0))
    assert empty['summary']['rows']==0
    assert empty['completeness'].AvailablePercent.isna().all()


def test_report_page_interactions():
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    root=Path(__file__).resolve().parents[1]
    app=AppTest.from_file(str(root/'pages/15_Cutting_Performance.py'),default_timeout=60).run()
    app.button[0].click().run(timeout=60)
    assert not app.exception
    assert len(app.metric)==4
    assert 'cutting_performance_result' in app.session_state
    quality=AppTest.from_file(str(root/'pages/16_Dataset_Quality.py'),default_timeout=60).run()
    assert not quality.exception
    assert len(quality.metric)==3
    assert not any(h.value == 'Volume consistency' for h in quality.subheader)
