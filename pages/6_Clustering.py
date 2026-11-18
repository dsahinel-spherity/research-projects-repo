import plotly.express as px
import streamlit as st
from src.ui import context, chart
from src.ml.clustering import segment
from src.preprocessing.feature_engineering import NUMERIC_FEATURES
st.title('Clustering & Segmentation')
c = context()
with st.form('clusters'):
    algorithm = st.selectbox('Clustering algorithm', ['K-Means', 'DBSCAN'])
    options = [f for f in NUMERIC_FEATURES if c['logs'][f].notna().any()]
    features = st.multiselect('Segmentation measurements', options, default=[f for f in ['LogLength', 'MidDiameter_ob', c['volume'], 'DBH'] if f in options])
    a,b,z = st.columns(3)
    k = a.slider('K-Means clusters', 2, 10, 3)
    eps = b.slider('DBSCAN radius (standardized units)', .1, 3., .7, .1)
    minimum = z.slider('DBSCAN minimum samples', 2, 30, 5)
    run = st.form_submit_button('Segment observations', type='primary')
if run:
    try:
        st.session_state.cluster_result = segment(c['logs'], features, algorithm, k, eps, minimum)
    except ValueError as exc:
        st.error(str(exc))
if 'cluster_result' in st.session_state:
    d, profiles, score, descriptions, variance = st.session_state.cluster_result
    st.metric('Silhouette score (excluding DBSCAN noise)', 'Not applicable' if score is None else f'{score:.3f}')
    st.caption(f'PCA displays {100*sum(variance):.1f}% of standardized feature variance. Missing measurements are median-imputed; rows missing all selected measurements are excluded. Clusters are fitted in the full standardized feature space.')
    chart(px.scatter(d, x='PC1', y='PC2', color='Cluster', hover_data=['StemKey', 'LogKey']))
    chart(px.histogram(d, x='Cluster', title='Cluster distribution · −1 denotes DBSCAN noise'))
    st.dataframe(profiles, use_container_width=True)
    for description in descriptions.values():
        st.write(description)
    st.caption('Profiles use recorded measurements in their original units. These are measured segments, not verified commercial quality classes. Silhouette measures geometric separation, not business usefulness.')
    st.download_button('Download segments', d.to_csv(index=False), 'segments.csv', 'text/csv')
