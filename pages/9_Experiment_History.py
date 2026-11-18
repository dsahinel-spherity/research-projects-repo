import json
import streamlit as st
from src.ui import context, ROOT
from src.ml.evaluation import history
st.title('ML Experiment History')
context()
records, errors = history(ROOT / 'experiments')
st.caption('Local JSON records retain dataset hash, recovery status, filters, seed, configuration, selected features, split tree IDs, cross-validation and held-out metrics. Fitted models remain in this session; no untrusted pickle files are loaded.')
for error in errors:
    st.warning(error)
if not records:
    st.info('No saved experiments yet. Train a model to create the first local record.')
else:
    chosen = st.selectbox('Saved experiment', range(len(records)), format_func=lambda i: records[i]['id'] + ' · ' + records[i]['config']['algorithm'] + ' → ' + records[i]['config']['target'])
    st.json(records[chosen])
    st.download_button('Download experiment metadata', json.dumps(records[chosen], indent=2), records[chosen]['id']+'.json', 'application/json')
