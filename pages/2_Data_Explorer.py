import streamlit as st
from src.ui import context
st.title('Data Explorer')
c = context()
st.markdown('#### Ingestion and relational validation')
st.json(c['dataset'].report, expanded=False)
st.write('Values excluded from analytic copies', {k:v for k,v in c['invalid'].items() if v})
for label, frame in [('Filtered trees', c['trees']), ('Filtered joined logs', c['logs']), ('Raw parsed trees', c['dataset'].trees), ('Raw parsed logs', c['dataset'].logs)]:
    with st.expander(label, expanded=label == 'Filtered joined logs'):
        st.dataframe(frame, use_container_width=True)
        st.download_button('Download ' + label, frame.to_csv(index=False), label.replace(' ', '_') + '.csv', 'text/csv')
        st.dataframe(frame.isna().sum().rename('Missing values').to_frame(), use_container_width=True)
        st.dataframe(frame.describe(include='number').T, use_container_width=True)
st.caption('LogKey is local to each tree. The relational key is MachineID + StemKey + LogKey; TreeID combines machine and StemKey. Missing and ambiguous measurements remain missing. Raw parsed exports retain invalid ranges before analytic exclusions.')
