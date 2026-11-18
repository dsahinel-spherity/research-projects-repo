import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from src.ui import context, chart, number
from src.explainability.shap_engine import permutation, explain
st.title('Explainable AI')
context()
if 'active_experiment' not in st.session_state:
    st.info('Train a model on Predictive Analytics to explain its held-out predictions.')
    st.stop()
r = st.session_state.active_experiment
st.write(f"Active model: {r['config'].algorithm} → {r['config'].target} · {r['config'].mode}")
st.caption('Explanations describe model associations, not physical causation. Correlated features may share or obscure attribution.')
if 'permutation' not in r:
    with st.spinner('Measuring held-out permutation importance…'):
        r['permutation'] = permutation(r)
importance = r['permutation']
chart(px.bar(importance, x='MAE increase', y='Feature', orientation='h', error_x='Std', title='Permutation importance · held-out MAE increase'))
st.caption('Negative importance means shuffling improved this sample’s MAE. Uncertainty bars are standard deviations over five shuffles, not confidence intervals.')
row = st.selectbox('Held-out log', list(range(len(r['heldout']))), format_func=lambda i: f"Tree {r['heldout'].iloc[i].StemKey} · log {r['heldout'].iloc[i].LogKey}")
record = r['heldout'].iloc[row]
for col, key in zip(st.columns(3), ['Actual', 'Predicted', 'Residual']):
    col.metric(key, number(record[key], 4))
chart(px.scatter(r['heldout'], x='Actual', y='Predicted', title='Held-out actual versus predicted'))
if st.button('Compute bounded SHAP explanations', type='primary'):
    try:
        with st.spinner('Explaining up to 50 held-out records using training-only background…'):
            values, positions = explain(r, selected_position=row)
            st.session_state.shap_result = (r['metadata']['id'], row, values, positions)
    except Exception as exc:
        st.warning(f'SHAP unavailable for this run: {exc}. Permutation importance remains available above.')
cache = st.session_state.get('shap_result')
if cache and cache[0] == r['metadata']['id'] and cache[1] == row:
    import shap
    import matplotlib.pyplot as plt
    values = cache[2]
    mean_abs = pd.DataFrame({'Feature': values.feature_names, 'Mean |SHAP|': np.abs(values.values).mean(axis=0)})
    chart(px.bar(mean_abs.sort_values('Mean |SHAP|'), x='Mean |SHAP|', y='Feature', orientation='h', title='Global importance · sampled held-out logs'))
    shap.plots.beeswarm(values, show=False, max_display=15)
    st.pyplot(plt.gcf(), clear_figure=True)
    plt.close('all')
    shap.plots.waterfall(values[0], show=False, max_display=12)
    st.pyplot(plt.gcf(), clear_figure=True)
    plt.close('all')
    local = pd.DataFrame({'Feature': values.feature_names, 'Contribution': values.values[0]})
    local = local.iloc[np.argsort(-np.abs(local.Contribution))]
    st.dataframe(local, use_container_width=True)
    st.write('For this prediction, the largest absolute model contributions are associated with ' + ', '.join(local.Feature.head(3)) + '. Positive contributions increase the prediction relative to the training-background expectation; negative contributions decrease it.')
else:
    st.caption('Compute SHAP for the selected log to see local contributions and the waterfall. Encoded categories are shown separately; SHAP uses a bounded permutation approximation.')
