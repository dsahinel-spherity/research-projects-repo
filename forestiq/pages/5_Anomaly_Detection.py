import pandas as pd
import plotly.express as px
import streamlit as st
from src.ui import context, chart
from src.preprocessing.feature_engineering import NUMERIC_FEATURES
from src.ml.anomaly_detection import detect
st.title('Anomaly Detection')
c = context()
with st.form('anomalies'):
    options = [f for f in NUMERIC_FEATURES if c['logs'][f].notna().any()]
    features = st.multiselect('Isolation Forest measurements', options, default=[f for f in ['LogLength', 'MidDiameter_ob', c['volume'], 'DBH', 'CuttingDuration'] if f in options])
    contamination = st.slider('Expected anomaly proportion', .01, .3, .05, .01)
    run = st.form_submit_button('Detect unusual records', type='primary')
if run:
    try:
        st.session_state.anomaly_result = (detect(c['logs'], features, contamination), features)
    except ValueError as exc:
        st.error(str(exc))
if 'anomaly_result' in st.session_state:
    d, selected = st.session_state.anomaly_result
    for col, label, value in zip(st.columns(3), ['Logs analyzed', 'Flagged anomalies', 'Flagged percentage'], [len(d), int(d.Anomaly.sum()), f'{100*d.Anomaly.mean():.1f}%']):
        col.metric(label, value)
    st.caption(f"{d.attrs.get('excluded_all_missing', 0)} all-missing rows excluded. Partial missing measurements are median-imputed for the detector only.")
    st.warning('Unusual does not mean faulty. Short logs, manual cuts and zero price-volume records may be legitimate. The contamination setting influences how many records are flagged.')
    x = st.selectbox('Scatter X', selected)
    y = st.selectbox('Scatter Y', selected, index=min(1, len(selected)-1))
    chart(px.scatter(d.assign(Status=d.Anomaly.map({True:'Flagged', False:'Typical'})), x=x, y=y, color='Status', hover_data=['StemKey', 'LogKey', 'AnomalyScore']))
    chart(px.histogram(d, x='AnomalyScore', color=d.Anomaly.astype(str), title='Anomaly scores · higher is more unusual'))
    feature = st.selectbox('Compare feature distributions', selected)
    chart(px.box(d, x=d.Anomaly.astype(str), y=feature, title='Observed measurements by anomaly flag'))
    st.markdown('#### Most unusual records')
    cols = list(dict.fromkeys(['StemKey', 'LogKey', 'CuttingReason'] + selected + ['AnomalyScore', 'UnusualFeatures']))
    st.dataframe(d.loc[d.Anomaly, cols], use_container_width=True)
    st.caption('Descriptions compare observed values with medians in standard-deviation units. These are descriptive deviations, not explanations of the Isolation Forest’s internal decision path.')
    st.markdown('#### Automatic and manual cutting categories')
    category = d.groupby('CuttingReason', dropna=False).Anomaly.agg(Analyzed='size', Flagged='sum', Rate='mean').reset_index()
    st.dataframe(category, use_container_width=True)
    chart(px.bar(category, x='CuttingReason', y='Rate', title='Flag rate by recorded cutting reason'))
    st.caption('Categories retain their source labels. Rates describe this filtered sample and do not establish machine faults or differences in process quality.')
    st.download_button('Download anomaly results', d.to_csv(index=False), 'anomalies.csv', 'text/csv')
