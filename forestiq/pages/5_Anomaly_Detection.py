import plotly.express as px
import streamlit as st
from src.ui import context, chart
from src.preprocessing.feature_engineering import NUMERIC_FEATURES
from src.ml.anomaly_detection import detect, detect_autoencoder

st.title('Anomaly Detection')
st.caption('Run Isolation Forest or a neural Autoencoder independently. Both learn from the current logs, assuming most are normal. Flags are review candidates, not confirmed errors.')
c = context()
frame = c['logs']
options = [f for f in NUMERIC_FEATURES if frame[f].notna().any()]


def field_label(field):
    labels = {'LogLength': 'Length (cm)', 'Volume_price': 'Price volume (m³)', 'Volume_ob': 'Over-bark volume (m³)',
              'Volume_ub': 'Under-bark volume (m³)', 'CuttingDuration': 'Cutting duration (s)',
              'TreeID': 'Tree ID', 'LogKey': 'Log number', 'ProductKey': 'Product code', 'LogscomID': 'Logscom ID',
              'AnomalyScore': 'Anomaly score', 'UnusualFeatures': 'Measurement to review',
              'ScoringStatus': 'Data status', 'MissingMeasurements': 'Missing input count'}
    if field in labels:
        return labels[field]
    if 'Diameter' in field or field == 'DBH':
        return field.replace('_ob', ' over bark').replace('_ub', ' under bark') + ' (mm)'
    return field


def render_results(d, key):
    for col, label, value in zip(st.columns(3), ['Logs', 'Flagged for review', 'Unscorable logs'], [len(d), int(d.Anomaly.sum()), int(d.AnomalyScore.isna().sum())]):
        col.metric(label, value)
    st.caption(f"Saved run: {', '.join(d.attrs['features'])}; requested anomaly proportion {d.attrs['fraction']:.0%}. Run again to apply changed controls.")
    st.subheader('Flagged logs')
    flagged = d.loc[d.Anomaly].sort_values('AnomalyScore', ascending=False)
    fields = list(dict.fromkeys([f for f in ['TreeID', 'LogKey', 'LogscomID', 'ProductKey'] if f in d] + ['AnomalyScore', 'UnusualFeatures', 'ScoringStatus', 'MissingMeasurements'] + d.attrs['features']))
    def review_table(rows):
        table = rows[fields].copy()
        table['UnusualFeatures'] = table.UnusualFeatures.map(lambda f: field_label(f) if isinstance(f, str) else None)
        return table.rename(columns=field_label)
    st.caption('Highest scores first. Scores are specific to each method, not fault probabilities. ' + d.attrs['explanation'] + '.')
    if flagged.empty:
        st.info('No logs were flagged in this run.')
    else:
        table = review_table(flagged)
        table.insert(0, 'Review order', range(1, len(table)+1))
        config = {'Anomaly score': st.column_config.NumberColumn(format='%.6f')}
        for f in d.attrs['features']:
            config[field_label(f)] = st.column_config.NumberColumn(format='%.4f' if f.startswith('Volume') else '%.2f')
        st.dataframe(table, column_config=config, use_container_width=True, hide_index=True)
        st.download_button('Download flagged logs', table.to_csv(index=False), key+'_flagged_logs.csv', 'text/csv', key=key+'_flagged_download')
    unscorable = d.loc[d.AnomalyScore.isna()]
    if len(unscorable):
        with st.expander(f'Unscorable logs requiring data review ({len(unscorable)})'):
            st.dataframe(review_table(unscorable), hide_index=True)
    status = d.assign(Status=d.Anomaly.map({True:'Flagged', False:'Not flagged'}))
    chart(px.scatter(status, x='PC1', y='PC2', color='Status', hover_data=['TreeID','LogKey','AnomalyScore'], title=key.replace('_',' ').title()+' · PCA 2-D view'))
    st.caption('PCA only displays records; it is not a detector. Partial missing inputs are median-imputed and marked. Negative or entirely missing inputs are unscorable, not normal. Imputed inputs cannot be named as the measurement to review.')
    with st.expander('All scored records'):
        st.dataframe(d, use_container_width=True, hide_index=True)
    st.download_button('Download scored records', d.to_csv(index=False), key+'_results.csv', 'text/csv', key=key+'_all_download')


for tab, method, key in zip(st.tabs(['Isolation Forest', 'Autoencoder']), ['Isolation Forest', 'Autoencoder'], ['isolation_forest', 'autoencoder']):
    with tab:
        if method == 'Autoencoder':
            st.caption('A small neural network learns to reconstruct measurements through a smaller hidden layer. Larger reconstruction errors are more unusual. The top requested percentage is flagged even if the dataset is clean. This is in-sample screening, not proof of accuracy on new logs.')
        features = st.multiselect('Measurements', options, default=[f for f in ['LogLength', 'MidDiameter_ob', c['volume']] if f in options], key=key+'_features')
        fraction = st.slider('Expected proportion of unusual logs (%)', 1, 30, 5, key=key+'_fraction') / 100
        bottleneck = 1
        if method == 'Autoencoder':
            bottleneck = int(st.number_input('Hidden bottleneck size', min_value=1, max_value=max(1, len(features)-1), value=1, key='autoencoder_bottleneck'))
        if st.button('Run '+method, type='primary', key=key+'_run'):
            st.session_state.pop(key+'_result', None)
            try:
                with st.spinner('Running '+method+'…'):
                    d = detect(frame, features, fraction) if method == 'Isolation Forest' else detect_autoencoder(frame, features, fraction, bottleneck=bottleneck)
                    st.session_state[key+'_result'] = (c['signature'], d)
            except ValueError as exc:
                st.info(str(exc))
        result = st.session_state.get(key+'_result')
        if result and result[0] == c['signature']:
            if 'features' in result[1].attrs:
                render_results(result[1], key)
            else:
                st.info('Run the model again to refresh results from the previous page version.')
