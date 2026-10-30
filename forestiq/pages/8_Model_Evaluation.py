import pandas as pd
import streamlit as st
from src.ui import context, ROOT, evaluation_view
from src.ml.evaluation import history
st.title('Model Evaluation')
c = context()
if 'active_experiment' in st.session_state:
    evaluation_view(st.session_state.active_experiment)
records, errors = history(ROOT / 'experiments')
matching = [r for r in records if r.get('scope_signature') == c['signature']]
st.markdown('#### Saved runs for this data scope')
if matching:
    comparisons = []
    for r in matching:
        cfg = r['config']
        comparisons.append({'Experiment': r['id'], 'Algorithm': cfg['algorithm'], 'Target': cfg['target'], 'Mode': cfg['mode'], 'Temporal': cfg['temporal'], 'Seed': cfg['seed'], 'Test fraction': cfg['test_size'], **r['test'], 'Baseline MAE': r['baseline']['MAE']})
    st.dataframe(pd.DataFrame(comparisons), use_container_width=True)
    st.caption('Compare only runs with the same target, mode and held-out tree IDs. Repeatedly selecting models against the same test set biases evaluation; reserve a fresh operation for final validation.')
else:
    st.info('Train models on Predictive Analytics to evaluate them here.')
for error in errors:
    st.warning(error)
