import plotly.express as px
import streamlit as st
from src.ui import context, chart, number
from src.ml.anomaly_detection import detect
st.title('Production analytics')
st.caption('The essential trends and checks for understanding your timber production.')
c = context()
d, v = c['logs'], c['volume']
if d.empty:
    st.info('No processed logs are recorded in this scope. Tree and area information remains available in Overview and Timber Data.')
    st.stop()
production, quality = st.tabs(['Production', 'Data quality'])
with production:
    a,b = st.columns(2)
    with a:
        timed = d.dropna(subset=['FellingTime']).assign(Hour=lambda x: x.FellingTime.dt.floor('h')).groupby('Hour')[v].sum(min_count=1).reset_index()
        chart(px.bar(timed, x='Hour', y=v, title='Recorded volume by felling hour · UTC', labels={v:'Volume · m³'}))
    with b:
        chart(px.histogram(d, x='LogLength', title='Log length distribution · cm'))
    st.subheader('Product summary')
    summary = d.groupby('ProductKey', dropna=False).agg(Logs=('LogKey','size'), RecordedVolumes=(v,'count'), Volume_m3=(v, lambda x:x.sum(min_count=1)), AverageLength_cm=('LogLength','mean')).reset_index()
    st.dataframe(summary, use_container_width=True, hide_index=True)
    observed = summary.dropna(subset=['Volume_m3'])
    if len(observed):
        top = observed.loc[observed.Volume_m3.idxmax()]
        st.write(f'Product {top.ProductKey} has the highest recorded volume in this scope: {top.Volume_m3:.2f} m³.')
with quality:
    fields = ['LogLength', v, 'FellingLatitude', 'FellingTime', 'DropOffLatitude', 'DropOffTime']
    st.dataframe(d[fields].isna().sum().rename('Missing records').to_frame(), use_container_width=True)
    st.caption('Missing locations and timestamps are common in standard HPR files; detailed placement events require machine extensions. Zero price volume and manual cuts are not automatically errors.')
    if st.button('Check for unusual measurements'):
        features = [f for f in ['LogLength','MidDiameter_ob',v] if d[f].notna().any()]
        try:
            result = detect(d, features, contamination=.05)
            st.session_state.basic_anomalies = (c['signature'], v, result)
        except ValueError as exc:
            st.info(str(exc))
    cached = st.session_state.get('basic_anomalies')
    if cached and cached[:2] == (c['signature'], v):
        flagged = cached[2][cached[2].Anomaly]
        st.write(f'{len(flagged)} of {len(cached[2])} logs flagged for review by Isolation Forest (5% contamination setting).')
        st.dataframe(flagged[['StemKey','LogKey','LogLength',v,'CuttingReason','UnusualFeatures']], use_container_width=True, hide_index=True)
        st.caption('Flags identify uncommon measurements, not confirmed faults. Descriptions compare observed values with the sample distribution.')
