"""Shared application state, filtered dataset context and chart presentation."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from src.ingestion.stanford_parser import parse_hpr
from src.preprocessing.feature_engineering import filter_data
from src.preprocessing.timber import prepare_timber
from src.datasets.registry import list_datasets

ROOT = Path(__file__).resolve().parents[1]
VOLUME_LABELS = {'Price volume (m³)': 'Volume_price', 'Over-bark volume (m³)': 'Volume_ob', 'Under-bark volume (m³)': 'Volume_ub'}
PARSER_REVISION = 'timber_hpr_v4'

@st.cache_data(show_spinner=False, max_entries=8)
def _cached_parse(raw, recover, parser_revision):
    return parse_hpr(source=raw, recover=recover)


def load_data(raw, recover):
    """Reject pre-timber cached schemas and rebuild them from the original XML."""
    dataset = _cached_parse(raw, recover, PARSER_REVISION)
    required_trees = {'ObjectKey', 'CraneCRS', 'CraneReceiverPosition', 'CraneCoordinateDate'}
    required_logs = {'ProductName', 'LogCRS', 'LogReceiverPosition', 'LogCoordinateDate'}
    if (not required_trees.issubset(dataset.trees.columns)
            or not required_logs.issubset(dataset.logs.columns)
            or 'schema' not in dataset.report):
        _cached_parse.clear()
        dataset = parse_hpr(source=raw, recover=recover)
    return dataset


def chart(fig):
    fig.update_layout(template='plotly_white', margin=dict(l=15, r=15, t=45, b=20), colorway=['#140A9A', '#E5B541', '#7089B8', '#C66860'], font=dict(family='Arial', color='#20243C'), paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)


def number(value, digits=2):
    return 'Not recorded' if pd.isna(value) else f'{value:,.{digits}f}'


def _remember(key):
    # This copy is not owned by a widget, so Streamlit's page cleanup retains it.
    st.session_state['_saved_' + key] = st.session_state[key]


def _restore(key, default):
    saved_key = '_saved_' + key
    if saved_key not in st.session_state:
        # Adopt an existing selection when upgrading an already-open session.
        st.session_state[saved_key] = st.session_state.get(key, default)
    st.session_state[key] = st.session_state[saved_key]


def _choose_dataset():
    _remember('source_dataset')
    _clear_upload()


def _clear_upload():
    st.session_state.pop('_uploaded_source', None)
    st.session_state['_upload_generation'] = st.session_state.get('_upload_generation', 0) + 1


def _remember_upload(key):
    upload = st.session_state.get(key)
    st.session_state['_uploaded_source'] = {'raw': upload.getvalue(), 'name': upload.name} if upload else None


def _filter_widget(title, col, frame):
    key = 'filter_' + col
    options = sorted(frame[col].dropna().unique().tolist())
    _restore(key, [])
    st.session_state[key] = [value for value in st.session_state[key] if value in options]
    return st.multiselect(title, options, key=key, on_change=_remember, args=(key,))


def context():
    with st.sidebar:
        st.markdown('### Log-Chain')
        st.caption('Timber production analytics')
        registered, registry_errors = list_datasets(ROOT / 'data/registry')
        choices = ['Bundled sample'] + [r['id'] for r in registered]
        names = {r['id']: r['name'] for r in registered}
        _restore('source_dataset', 'Bundled sample')
        if st.session_state.source_dataset not in choices:
            st.session_state.source_dataset = 'Bundled sample'
        choice = st.selectbox('Dataset', choices, format_func=lambda x: names.get(x, x), key='source_dataset', on_change=_choose_dataset)
        for error in registry_errors:
            st.warning('Registry entry unavailable: ' + error)
        upload_key = 'source_upload_' + str(st.session_state.get('_upload_generation', 0))
        st.file_uploader('Upload StanForD production', type=['txt', 'xml', 'hpr'], key=upload_key, on_change=_remember_upload, args=(upload_key,))
        upload = st.session_state.get('_uploaded_source')
        if upload:
            st.caption('Active upload: ' + upload['name'])
            st.button('Clear uploaded file', on_click=_clear_upload)
        files = sorted((ROOT / 'data/raw').glob('*')) + sorted(ROOT.glob('25119*.txt'))
        files = [p for p in files if p.suffix.lower() in ['.txt', '.xml', '.hpr']]
        if upload:
            raw, source = upload['raw'], upload['name']
        elif choice != 'Bundled sample':
            try:
                raw = (ROOT / 'data/registry' / choice / 'source.hpr').read_bytes()
                source = names[choice]
            except OSError as exc:
                st.error(f'Registered source unavailable: {exc}')
                st.stop()
        elif files:
            source = files[0].name
            raw = files[0].read_bytes()
        else:
            st.info('Upload an HPR dataset to begin.')
            st.stop()
        default_recovery = upload is None and (choice == 'Bundled sample' or next(r for r in registered if r['id'] == choice)['recovery_enabled'])
        with st.expander('Import options'):
            recovery_key = 'recovery_' + ('upload_' if upload else choice + '_') + hashlib.sha256(raw).hexdigest()[:12]
            _restore(recovery_key, default_recovery)
            recovery = st.checkbox('Recover complete trees from malformed XML', key=recovery_key, on_change=_remember, args=(recovery_key,), help='Explicit salvage: excludes fragments with no intact tree context. Warnings remain visible.')
            st.caption(source)
    try:
        dataset = load_data(raw, recovery)
    except (ValueError, UnicodeError) as exc:
        st.error(str(exc))
        st.stop()
    if dataset.report['recovered']:
        st.warning('Recovered sample: incomplete opening records were excluded. Totals cover the intact trees only.')
    elif dataset.report['schema']['status'] == 'invalid':
        st.warning('The source does not pass the official HPR schema check. Review source validation details before relying on it.')
    if dataset.report['warnings'] or dataset.report['schema']['status'] == 'invalid':
        with st.expander('Source validation details'):
            for warning in dataset.report['warnings']:
                st.write(warning)
            st.json(dataset.report['schema'])
    if not dataset.report['analytics_safe']:
        st.error('Relational validation failed. Analytics are blocked to prevent duplicate counting. Correct the source IDs and re-upload.')
        st.json(dataset.report)
        st.dataframe(dataset.trees)
        st.dataframe(dataset.logs)
        st.stop()
    trees, logs, invalid = prepare_timber(dataset)
    if sum(invalid.values()):
        st.warning(f'{sum(invalid.values())} negative measurements/durations or invalid coordinates excluded from analytics; raw values remain available in Data Explorer.')
    if dataset.report['invalid_numeric_values'] or dataset.report['invalid_timestamps']:
        st.warning('Unparseable values were marked missing. Inspect the ingestion report in Data Explorer.')
    with st.sidebar:
        st.divider()
        st.markdown('#### Production scope')
        selected = {}
        for col, title in [('HarvestObjectID', 'Harvest object'), ('ProductKey', 'Product category')]:
            table = trees if col == 'HarvestObjectID' else logs
            selected[col] = _filter_widget(title, col, table)
        with st.expander('More filters'):
            for col, title in [('SpeciesGroupKey', 'Species group'), ('CuttingReason', 'Cutting reason'), ('OperatorKey', 'Operator')]:
                selected[col] = _filter_widget(title, col, logs)
        dates = None
        valid_dates = trees.HarvestDate.dropna()
        if not valid_dates.empty:
            _restore('limit_dates', False)
            if st.checkbox('Limit harvest dates (UTC)', key='limit_dates', on_change=_remember, args=('limit_dates',)):
                date_key = 'date_range_' + dataset.fingerprint[:12]
                _restore(date_key, (valid_dates.min().date(), valid_dates.max().date()))
                bounds = st.date_input('Harvest date range', key=date_key, on_change=_remember, args=(date_key,))
                if isinstance(bounds, (tuple, list)) and len(bounds) == 2:
                    dates = bounds
                else:
                    st.info('Select both ends of the date range.')
                    st.stop()
        _restore('volume_definition', next(iter(VOLUME_LABELS)))
        volume_label = st.selectbox('Recorded volume definition', list(VOLUME_LABELS), key='volume_definition', on_change=_remember, args=('volume_definition',))
        volume = VOLUME_LABELS[volume_label]
    trees, logs = filter_data(trees, logs, selected, dates)
    if trees.empty:
        st.info('No trees match this scope. Clear or broaden the sidebar filters.')
        st.stop()
    signature = hashlib.sha256((PARSER_REVISION + dataset.fingerprint + str(recovery) + json.dumps(selected, sort_keys=True) + str(dates)).encode()).hexdigest()
    if st.session_state.get('scope_signature') != signature:
        for key in ['active_experiment', 'shap_result', 'anomaly_result', 'isolation_forest_result', 'autoencoder_result', 'cluster_result', 'cut_classifier_result', 'feature_selection_result']:
            st.session_state.pop(key, None)
        st.session_state.scope_signature = signature
    st.caption(f'{len(trees):,} distinct trees · {len(logs):,} logs in scope · {volume_label} · timestamps UTC · diameters mm · lengths cm')
    return {'dataset': dataset, 'trees': trees, 'logs': logs, 'volume': volume, 'volume_label': volume_label, 'signature': signature, 'source': source, 'invalid': invalid, 'filters': selected, 'dates': str(dates), 'raw': raw, 'recovery': recovery}


def evaluation_view(result):
    m = result['metadata']
    cols = st.columns(4)
    for col, name in zip(cols[:3], ['MAE', 'RMSE', 'R2']):
        col.metric('Test ' + name, number(m['test'][name]))
    cols[3].metric('Mean baseline MAE', number(m['baseline']['MAE']))
    units = 'cm' if result['config'].target == 'LogLength' else ('seconds' if result['config'].target == 'CuttingDuration' else 'm³')
    st.write(f"The model's held-out MAE is {m['test']['MAE']:.3f} {units}, evaluated on {m['test_rows']} logs from {m['test_trees']} unseen trees.")
    if m['test']['MAE'] >= m['baseline']['MAE']:
        st.warning('This model does not improve on the training-mean baseline by test MAE. Useful predictive performance is not demonstrated.')
    else:
        st.info('Test MAE improves on the training-mean baseline for this split. This does not establish performance on other sites, species or seasons.')
    st.dataframe(pd.DataFrame({'Train': m['train'], 'Test': m['test'], 'Mean baseline': m['baseline']}).T, use_container_width=True)
    if m['test']['MAE'] > 1.5 * max(m['train']['MAE'], 1e-12):
        st.warning('Test error is substantially larger than training error; inspect overfitting and distribution shifts.')
    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(result['heldout'], x='Actual', y='Predicted', hover_data=['StemKey', 'LogKey'], title='Held-out observations')
        low, high = result['heldout'].Actual.min(), result['heldout'].Actual.max()
        fig.add_shape(type='line', x0=low, x1=high, y0=low, y1=high, line=dict(dash='dash', color='gray'))
        chart(fig)
    with c2:
        chart(px.histogram(result['heldout'], x='Residual', title='Residuals · actual − predicted'))
    st.markdown('#### Training-partition cross-validation')
    cv = pd.DataFrame(m['cv'])
    st.dataframe(cv, use_container_width=True)
    st.dataframe(cv.agg(['mean', 'std']), use_container_width=True)
    st.caption('Preprocessing is refitted inside every fold. The held-out partition is excluded from cross-validation. R² is undefined for constant targets or fewer than two observations.')
    if 'geometric_baseline' in m:
        st.write('Frustum baseline on the same held-out records', m['geometric_baseline'])
        st.caption('V = πL(Db² + DbDt + Dt²)/12, with cm → m and mm → m conversion. This geometric approximation is not the harvester’s price-volume rule.')
