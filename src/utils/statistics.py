import numpy as np
import pandas as pd
from scipy import stats


def associations(frame, x, y):
    pairs = frame[[x, y]].dropna()
    if x == y or len(pairs) < 3 or pairs[x].nunique() < 2 or pairs[y].nunique() < 2:
        raise ValueError('Choose distinct, varying variables with at least three complete pairs.')
    return pd.DataFrame([{'Method': name, 'Coefficient': float(value.statistic), 'p-value': float(value.pvalue), 'Pairs': len(pairs)} for name, value in [('Pearson', stats.pearsonr(pairs[x], pairs[y])), ('Spearman', stats.spearmanr(pairs[x], pairs[y]))]])


def tree_mean_ci(frame, variable, seed=42):
    """Bootstrap trees, with each tree contributing its mean (not each log)."""
    values = frame.groupby('TreeID')[variable].mean().dropna().to_numpy()
    if len(values) < 5:
        raise ValueError('At least five trees with measurements are required for a descriptive interval.')
    rng = np.random.default_rng(seed)
    bootstrap = np.array([rng.choice(values, size=len(values), replace=True).mean() for _ in range(1000)])
    return {'tree_weighted_mean': float(values.mean()), 'low_95': float(np.quantile(bootstrap, .025)), 'high_95': float(np.quantile(bootstrap, .975)), 'trees': len(values)}


def outlier_summary(frame, columns):
    q1, q3 = frame[columns].quantile(.25), frame[columns].quantile(.75)
    iqr = q3 - q1
    return pd.DataFrame({'Observed': frame[columns].count(), 'Below 1.5 IQR': frame[columns].lt(q1-1.5*iqr).sum(), 'Above 1.5 IQR': frame[columns].gt(q3+1.5*iqr).sum()})
