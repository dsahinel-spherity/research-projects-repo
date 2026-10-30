import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def permutation(result):
    measured = permutation_importance(result['model'], result['X_test'], result['y_test'], scoring='neg_mean_absolute_error', n_repeats=5, random_state=result['config'].seed, n_jobs=1)
    return pd.DataFrame({'Feature': result['config'].features, 'MAE increase': measured.importances_mean, 'Std': measured.importances_std}).sort_values('MAE increase', ascending=False)


def explain(result, max_rows=50, selected_position=0):
    """Bounded permutation SHAP on the fitted transform; background is training-only."""
    import shap
    pipeline = result['model']
    prep, model = pipeline.named_steps['preprocess'], pipeline.named_steps['model']
    background = prep.transform(result['X_train'].sample(min(50, len(result['X_train'])), random_state=42))
    positions = [selected_position] + [i for i in range(min(max_rows, len(result['X_test']))) if i != selected_position]
    positions = positions[:max_rows]
    values = prep.transform(result['X_test'].iloc[positions])
    names = prep.get_feature_names_out()
    if len(names) > 100:
        raise ValueError('SHAP limited to 100 encoded features. Use permutation importance for this model.')
    explainer = shap.Explainer(model.predict, background, feature_names=names, algorithm='permutation', seed=42)
    explanation = explainer(values, max_evals=2 * len(names) + 1)
    return explanation, positions
