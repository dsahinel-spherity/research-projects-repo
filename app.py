from pathlib import Path
import streamlit as st
st.set_page_config(page_title='Log-Chain — Timber Analytics', layout='wide', initial_sidebar_state='expanded')
st.logo(str(Path(__file__).parent / 'assets/logchain.png'), size='large')
st.markdown('''<style>
[data-testid="stMetric"] {background:#ffffff;border:1px solid #e0e2ec;border-radius:10px;padding:16px;}
.block-container {padding-top:2rem;max-width:1450px;}
h1 {letter-spacing:-1px;} [data-testid="stSidebar"] {border-right:1px solid #e0e2ec;}
[data-testid="stSidebarHeader"] [data-testid="stLogo"] {height:180px !important;width:auto !important;max-height:180px !important;max-width:220px;object-fit:contain;}
</style>''', unsafe_allow_html=True)
advanced = st.sidebar.toggle('Show advanced tools', value=False, key='advanced_tools')
pages = [
    st.Page('pages/1_Overview.py', title='Overview', default=True),
    st.Page('pages/10_Timber_Data.py', title='Timber Data'),
    st.Page('pages/12_Log_Search.py', title='Log Search & Traceability'),
    st.Page('pages/11_Analytics.py', title='Analytics'),
    st.Page('pages/16_Dataset_Quality.py', title='Dataset Quality Report'),
]
if advanced:
    pages += [
        st.Page('pages/2_Data_Explorer.py', title='Data Explorer'),
        st.Page('pages/3_Predictive_Analytics.py', title='Predictive Analytics'),
        st.Page('pages/15_Cutting_Performance.py', title='Cutting Performance'),
        st.Page('pages/13_Cut_Classification.py', title='Logistic Regression'),
        st.Page('pages/14_Feature_Selection.py', title='Feature Selection'),
        st.Page('pages/4_Explainable_AI.py', title='Explainable AI'),
        st.Page('pages/5_Anomaly_Detection.py', title='Anomaly Detection'),
        st.Page('pages/6_Clustering.py', title='Clustering'),
        st.Page('pages/9_Experiment_History.py', title='Experiment History'),
    ]
st.navigation(pages).run()
