import numpy as np
import pandas as pd
from src.preprocessing.feature_engineering import frustum, filter_data, prepare


def test_frustum_units_and_missing():
    frame = pd.DataFrame({'LogLength': [100, np.nan], 'ButtDiameter_ob': [1000, 1000], 'TopDiameter_ob': [1000, 1000]})
    result = frustum(frame)
    assert np.isclose(result[0], np.pi / 4)
    assert np.isnan(result[1])


def test_duration_and_no_tree_double_count(dataset):
    trees, logs, invalid = prepare(dataset)
    assert len(trees) == 56 and len(logs) == 323
    assert logs.CuttingDuration.min() == 0
    assert logs.Volume_price.sum() > 0
    t, d = filter_data(trees, logs, {'CuttingReason': ['Other manual']})
    assert len(d) == 78
    assert t.TreeID.is_unique
    assert set(t.TreeID) == set(d.TreeID)


def test_negative_duration_excluded_without_changing_raw(dataset):
    import copy
    d = copy.deepcopy(dataset)
    d.logs.loc[0, 'CuttingEndTime'] = d.logs.loc[0, 'CuttingStartTime'] - pd.Timedelta(seconds=1)
    _, logs, invalid = prepare(d)
    assert pd.isna(logs.loc[0, 'CuttingDuration'])
    assert invalid['CuttingDuration'] == 1
