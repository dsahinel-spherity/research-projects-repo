"""Compare detectors on real KL logs. Reference checks are not verified error labels."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import math
import sys
import time
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ingestion.stanford_parser import parse_hpr
from src.preprocessing.feature_engineering import prepare, frustum
from src.ml.anomaly_detection import detect, detect_autoencoder


def main():
    root = Path(__file__).resolve().parents[1]
    source = next(root.glob('25119*.txt'))
    dataset = parse_hpr(source, recover=True)
    _, frame, invalid = prepare(dataset)
    features = ['LogLength', 'ButtDiameter_ob', 'TopDiameter_ob', 'Volume_ob']
    values = frame[features].replace([np.inf, -np.inf], np.nan)
    # An average is a descriptive reference only, not a definition of correctness.
    means, std = values.mean(), values.std(ddof=0).replace(0, np.nan)
    deviations = ((values-means)/std).abs()
    average_flag = deviations.gt(3).any(axis=1).astype('boolean')
    average_flag.loc[values.isna().any(axis=1)] = pd.NA
    volume = frustum(frame, 'ob')
    volume_flag = (frame.Volume_ob-volume).abs().gt(np.maximum(.01, .2*volume)).astype('boolean')
    volume_flag.loc[values.isna().any(axis=1) | (values <= 0).any(axis=1)] = pd.NA
    results, timing = {}, {}
    for method, function in [('Isolation Forest', detect), ('Autoencoder', detect_autoencoder)]:
        start = time.perf_counter()
        results[method] = function(frame, features, contamination=.05).reindex(frame.index)
        timing[method] = time.perf_counter()-start
    common = np.logical_and.reduce([d.AnomalyScore.notna().to_numpy() for d in results.values()])
    count = math.ceil(.05*common.sum())
    flagged = {}
    for method, d in results.items():
        # Match the review budget exactly; keep native app flags separately.
        indices = d.loc[common, 'AnomalyScore'].sort_values(ascending=False, kind='stable').head(count).index
        flagged[method] = pd.Series(frame.index.isin(indices), index=frame.index)
    def agreement(predicted, reference):
        usable = common & reference.notna()
        truth = reference.loc[usable].astype(bool)
        pred = predicted.loc[usable]
        overlap = int((pred & truth).sum())
        return {'comparable_logs': int(usable.sum()), 'reference_flags': int(truth.sum()),
                'overlap': overlap, 'model_only': int((pred & ~truth).sum()),
                'reference_only': int((~pred & truth).sum()),
                'fraction_of_model_flags_matching_reference': overlap/int(pred.sum()) if pred.sum() else None}
    both = flagged['Isolation Forest'] & flagged['Autoencoder']
    union = flagged['Isolation Forest'] | flagged['Autoencoder']
    report = {'dataset': source.name, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'logs': len(frame), 'trees': int(frame.TreeID.nunique()), 'features': features,
              'evaluation': 'In-sample exploratory comparison; no independently verified error labels; no accuracy winner can be determined.',
              'seed': 42, 'autoencoder_bottleneck': 1, 'review_fraction': .05, 'matched_flags_per_method': count,
              'reference_rules': {'average': 'Any selected measurement more than 3 population standard deviations from its dataset mean; incomplete rows unavailable.',
                                  'volume': 'Over-bark recorded volume differs from frustum estimate by more than max(0.01 m3, 20% of estimate); nonpositive/incomplete rows unavailable.'},
              'both_flagged': int(both.sum()), 'either_flagged': int(union.sum()),
              'flag_jaccard_overlap': float(both.sum()/union.sum()) if union.sum() else None,
              'preparation_invalid_counts': invalid,
              'versions': {x: importlib.metadata.version(x) for x in ['numpy', 'pandas', 'scikit-learn']},
              'methods': {m: {'native_app_flags': int(d.Anomaly.sum()), 'unscorable': int(d.AnomalyScore.isna().sum()),
                              'imputed': int(d.ScoringStatus.eq('Median-imputed').sum()), 'runtime_seconds': timing[m],
                              'average_reference': agreement(flagged[m], average_flag),
                              'volume_reference': agreement(flagged[m], volume_flag)} for m,d in results.items()}}
    table = frame[['TreeID','StemKey','LogKey','LogscomID','ProductKey']+features].copy()
    table['MaximumAbsoluteZScore'] = deviations.max(axis=1)
    table['AverageReferenceFlag'] = average_flag
    table['CalculatedVolume_m3'] = volume
    table['VolumeRelativeError'] = (frame.Volume_ob-volume).abs()/volume.replace(0, np.nan)
    table['VolumeReferenceFlag'] = volume_flag
    for method,d in results.items():
        prefix = method.replace(' ', '')
        table[prefix+'Score'] = d.AnomalyScore
        table[prefix+'Flag'] = flagged[method]
        table[prefix+'MeasurementToReview'] = d.UnusualFeatures
    out = root/'artifacts'/'anomaly_comparison_kl'
    out.mkdir(parents=True, exist_ok=True)
    (out/'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    table.to_csv(out/'all_logs_comparison.csv', index=False)
    table.loc[union].to_csv(out/'flagged_logs_comparison.csv', index=False)
    # All records, shuffled without scores/flags to reduce model-driven review bias.
    review = frame[['TreeID','LogKey','LogscomID','ProductKey']+features].sample(frac=1, random_state=42).copy()
    review['VerifiedStatus'] = 'unknown'
    review['ReviewEvidence'] = ''
    review.to_csv(out/'independent_review.csv', index=False)
    pd.DataFrame({'Mean': means, 'StandardDeviation': std, 'Minimum': values.min(), 'Maximum': values.max(), 'Missing': values.isna().sum()}).to_csv(out/'measurement_reference.csv')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
