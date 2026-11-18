"""Training-only feature selection; compare on common tree-group folds."""
from dataclasses import dataclass, asdict
import warnings
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_regression, f_classif, mutual_info_regression, mutual_info_classif, chi2
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from src.ml.evaluation import split_groups, metrics as regression_metrics
from src.ml.classification import cut_labels, metrics as classification_metrics
from src.preprocessing.feature_engineering import NUMERIC_FEATURES

CLASS_TARGET = 'Manual vs automatic'
TARGETS = ['CuttingDuration', 'LogLength', 'Volume_price', 'Volume_ob', 'Volume_ub', CLASS_TARGET]
METHODS = ['Correlation filter', 'F-test', 'Mutual information', 'Lasso', 'PCA', 'Chi-square']


def eligible_features(target):
    if target not in TARGETS:
        raise ValueError('Unsupported target.')
    excluded = {target}
    if target.startswith('Volume_'):
        excluded.update(['Volume_price', 'Volume_ob', 'Volume_ub'])
    return [c for c in NUMERIC_FEATURES if c not in excluded]


def applicable_methods(target):
    return [m for m in METHODS if m != ('Lasso' if target == CLASS_TARGET else 'Chi-square')]


@dataclass
class Config:
    target: str
    features: list[str]
    methods: list[str]
    sizes: list[int]
    seed: int = 42
    folds: int = 5
    correlation_limit: float = .9
    lasso_alpha: float = .01
    tolerance: float = .05
    algorithm: str = 'Feature Selection'


class Selector:
    """Fit imputation, scaling and ranking exclusively on the supplied training rows."""
    def __init__(self, method, size, config):
        self.method, self.size, self.config = method, size, config

    def fit(self, X, y):
        X = X.replace([np.inf, -np.inf], np.nan)
        self.columns = [c for c in X if X[c].nunique() > 1]
        if not self.columns:
            raise ValueError('No varying observed numeric predictors in a training fold.')
        self.imputer = SimpleImputer(strategy='median')
        raw = self.imputer.fit_transform(X[self.columns])
        self.scaler = StandardScaler().fit(raw)
        z = self.scaler.transform(raw)
        k = min(self.size, len(self.columns))
        self.pca = None
        self.scores = None
        classification = self.config.target == CLASS_TARGET
        if self.method == 'All inputs':
            chosen = list(range(len(self.columns)))
        elif self.method == 'PCA':
            self.pca = PCA(n_components=min(k, len(X)), svd_solver='full').fit(z)
            self.selected = [f'PC{i+1}' for i in range(self.pca.n_components_)]
            return self
        else:
            if self.method == 'Correlation filter':
                scores = np.abs(np.array([np.corrcoef(z[:, i], y)[0, 1] for i in range(z.shape[1])]))
            elif self.method == 'F-test':
                scores = (f_classif if classification else f_regression)(z, y)[0]
            elif self.method == 'Mutual information':
                scores = (mutual_info_classif if classification else mutual_info_regression)(z, y, random_state=self.config.seed)
            elif self.method == 'Lasso':
                if classification:
                    raise ValueError('Lasso selection is only used for numeric targets.')
                scaled_y = (np.asarray(y) - np.mean(y)) / np.std(y)
                scores = np.abs(Lasso(alpha=self.config.lasso_alpha, max_iter=20000).fit(z, scaled_y).coef_)
            elif self.method == 'Chi-square':
                if not classification:
                    raise ValueError('Chi-square needs a categorical target.')
                # Continuous measurements become nonnegative one-hot bin indicators.
                bins = KBinsDiscretizer(n_bins=5, encode='onehot-dense', strategy='quantile', subsample=None)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)
                    binned = bins.fit_transform(raw)
                bin_scores = np.nan_to_num(chi2(binned, y)[0], nan=0.)
                edges = np.r_[0, np.cumsum(bins.n_bins_)]
                scores = np.array([bin_scores[a:b].sum() for a,b in zip(edges[:-1], edges[1:])])
            else:
                raise ValueError('Unknown selection method.')
            scores = np.nan_to_num(scores, nan=0., posinf=np.finfo(float).max)
            self.scores = {c: float(s) for c,s in zip(self.columns, scores)}
            order = np.argsort(-scores, kind='stable')
            chosen = []
            for i in order:
                if self.method == 'Lasso' and scores[i] <= 1e-10:
                    continue
                if self.method == 'Correlation filter' and any(abs(np.corrcoef(z[:, i], z[:, j])[0, 1]) >= self.config.correlation_limit for j in chosen):
                    continue
                chosen.append(int(i))
                if len(chosen) == k:
                    break
            if not chosen:
                raise ValueError('Lasso retained no inputs; lower alpha or choose another method.')
        self.indices = chosen
        self.selected = [self.columns[i] for i in chosen]
        return self

    def transform(self, X):
        z = self.scaler.transform(self.imputer.transform(X[self.columns].replace([np.inf, -np.inf], np.nan)))
        return self.pca.transform(z) if self.pca is not None else z[:, self.indices]


def fit_candidate(X, y, method, size, config):
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        try:
            selector = Selector(method, size, config).fit(X, y)
            model = LogisticRegression(max_iter=2000) if config.target == CLASS_TARGET else Ridge(alpha=1.)
            model.fit(selector.transform(X), y)
        except ConvergenceWarning as exc:
            raise ValueError('Model failed to converge; adjust inputs or regularization.') from exc
    return selector, model


def evaluate(model, selector, X, y, classification):
    values = selector.transform(X)
    return classification_metrics(y, model.predict_proba(values)[:, 1]) if classification else regression_metrics(y, model.predict(values))


def compare(frame, config):
    allowed = eligible_features(config.target)
    if not config.features or not set(config.features) <= set(allowed):
        raise ValueError('Select eligible numeric inputs; target-derived fields and IDs are excluded.')
    if not config.methods or not set(config.methods) <= set(applicable_methods(config.target)):
        raise ValueError('Choose methods appropriate for this target.')
    if not config.sizes or any(not isinstance(k, int) or k < 1 or k > len(config.features) for k in config.sizes):
        raise ValueError('Subset sizes must be between 1 and the number of inputs.')
    if not 2 <= config.folds <= 8 or not 0 < config.correlation_limit <= 1 or not config.lasso_alpha > 0 or not 0 <= config.tolerance <= .5:
        raise ValueError('Invalid selection settings.')
    classification = config.target == CLASS_TARGET
    target = cut_labels(frame) if classification else frame[config.target].replace([np.inf, -np.inf], np.nan)
    mask = target.notna() & frame.TreeID.notna()
    data = frame.loc[mask].reset_index(drop=True)
    y = target.loc[mask].reset_index(drop=True).astype(int if classification else float)
    if len(data) < 15 or y.nunique() < 2:
        raise ValueError('Need at least 15 labelled rows and a varying target; classification requires both cut classes.')
    train, test = split_groups(data, .25, config.seed)
    X = data[config.features]
    Xtrain, ytrain = X.iloc[train], y.iloc[train]
    groups = data.iloc[train].TreeID
    k = min(config.folds, groups.nunique())
    if classification:
        if ytrain.nunique() != 2 or y.iloc[test].nunique() != 2:
            raise ValueError('Both classes must occur in training and test trees. Broaden filters or change seed.')
        k = min(k, int(ytrain.value_counts().min()))
        if k < 2:
            raise ValueError('Not enough minority-class rows for cross-validation.')
        folds = list(StratifiedGroupKFold(k, shuffle=True, random_state=config.seed).split(Xtrain, ytrain, groups))
        if any(ytrain.iloc[a].nunique() != 2 or ytrain.iloc[b].nunique() != 2 for a,b in folds):
            raise ValueError('Every CV fold needs both classes. Broaden scope or reduce folds.')
    else:
        folds = list(GroupKFold(k).split(Xtrain, ytrain, groups))
        if any(ytrain.iloc[a].nunique() < 2 for a,b in folds):
            raise ValueError('Target is constant in a training fold. Broaden scope.')
    metric = 'Balanced accuracy' if classification else 'MAE'
    candidates = [('All inputs', len(config.features))] + [(m,s) for m in config.methods for s in sorted(set(config.sizes))]
    rows, details, failures = [], {}, []
    for method, size in candidates:
        key = f'{method} / {size}'
        try:
            cv, selections = [], []
            for a,b in folds:
                selector, model = fit_candidate(Xtrain.iloc[a], ytrain.iloc[a], method, size, config)
                cv.append(evaluate(model, selector, Xtrain.iloc[b], ytrain.iloc[b], classification))
                selections.append(selector.selected)
            selector, model = fit_candidate(Xtrain, ytrain, method, size, config)
            scores = [m[metric] for m in cv]
            rows.append({'Candidate': key, 'Method': method, 'Requested size': size, 'Selected count': len(selector.selected),
                         'CV mean': float(np.mean(scores)), 'CV SD': float(np.std(scores)), 'Selected inputs / components': ', '.join(selector.selected)})
            details[key] = {'cv': cv, 'fold_selections': selections, 'selected': selector.selected, 'scores': selector.scores,
                            'pca_variance': selector.pca.explained_variance_ratio_.tolist() if selector.pca is not None else None,
                            'pca_loadings': selector.pca.components_.tolist() if selector.pca is not None else None,
                            'source_columns': selector.columns, '_selector': selector, '_model': model}
        except ValueError as exc:
            if method == 'All inputs':
                raise
            failures.append(f'{key}: {exc}')
    result = pd.DataFrame(rows)
    # Pick among ORIGINAL-column sets using training CV only. PCA is a separate comparison.
    eligible = result[result.Method != 'PCA']
    best = eligible['CV mean'].max() if classification else eligible['CV mean'].min()
    limit = best - config.tolerance if classification else best * (1 + config.tolerance)
    acceptable = eligible[eligible['CV mean'] >= limit] if classification else eligible[eligible['CV mean'] <= limit]
    recommended = acceptable.sort_values(['Selected count', 'CV mean'], ascending=[True, not classification]).iloc[0].Candidate
    # Only evaluate the chosen configuration and all-input reference on untouched test trees.
    for key in dict.fromkeys([recommended, f'All inputs / {len(config.features)}']):
        d = details[key]
        d['test'] = evaluate(d['_model'], d['_selector'], X.iloc[test], y.iloc[test], classification)
    for d in details.values():
        d.pop('_model'); d.pop('_selector')
    return {'config': asdict(config), 'metric': metric, 'recommendation': recommended, 'comparison': result.to_dict('records'),
            'details': details, 'failures': failures, 'excluded_rows': int((~mask).sum()), 'train_rows': len(train), 'test_rows': len(test),
            'train_group_ids': sorted(groups.unique().tolist()), 'test_group_ids': sorted(data.iloc[test].TreeID.unique().tolist()),
            'cv_groups': [{'train': sorted(groups.iloc[a].unique().tolist()), 'validation': sorted(groups.iloc[b].unique().tolist())} for a,b in folds]}
