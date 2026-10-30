import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


def segment(frame, features, algorithm='K-Means', k=3, eps=.7, min_samples=5, seed=42):
    if len(features) < 2 or len(frame) < max(5, k + 1):
        raise ValueError('Select at least two features and enough observations for clustering.')
    if any(frame[c].notna().sum() == 0 for c in features):
        raise ValueError('Remove entirely missing features.')
    result = frame.loc[frame[features].notna().any(axis=1)].copy()
    if len(result) < max(5, k + 1):
        raise ValueError('Too few rows with observed features.')
    X = StandardScaler().fit_transform(SimpleImputer(strategy='median').fit_transform(result[features]))
    if np.unique(X, axis=0).shape[0] < k and algorithm == 'K-Means':
        raise ValueError('Fewer distinct observations than requested clusters.')
    model = KMeans(n_clusters=k, n_init=10, random_state=seed) if algorithm == 'K-Means' else DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    result['Cluster'] = labels.astype(str)
    projection = PCA(n_components=2, random_state=seed)
    result[['PC1', 'PC2']] = projection.fit_transform(X)
    nonnoise = labels != -1
    unique = np.unique(labels[nonnoise])
    score = float(silhouette_score(X[nonnoise], labels[nonnoise], sample_size=min(2000, nonnoise.sum()), random_state=seed)) if 1 < len(unique) < nonnoise.sum() else None
    profiles = result.groupby('Cluster')[features].mean()
    global_mean, sd = result[features].mean(), result[features].std().replace(0, np.nan)
    descriptions = {}
    for cluster, row in profiles.iterrows():
        z = ((row - global_mean) / sd).dropna()
        top = z.abs().nlargest(2).index
        descriptions[cluster] = ('DBSCAN noise (not a quality class). ' if cluster == '-1' else f'Cluster {cluster}: ') + ', '.join(f'{c} averages {row[c]:.2f} ({"higher" if z[c] > 0 else "lower"} than the overall mean)' for c in top)
    return result, profiles, score, descriptions, projection.explained_variance_ratio_
