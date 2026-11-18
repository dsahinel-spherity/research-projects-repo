import streamlit as st
from src.ui import context, ROOT, evaluation_view
from src.ml.regression import Config, ALGORITHMS, eligible_features, run_experiment
from src.ml.evaluation import save_experiment
st.title('Predictive Analytics')
c = context()
experiment = st.selectbox('Experiment', ['Log length', 'Wood volume', 'Processing duration'])
target = {'Log length': 'LogLength', 'Wood volume': c['volume'], 'Processing duration': 'CuttingDuration'}[experiment]
st.caption('Target: ' + target)
mode = st.radio('Information available to the model', ['Retrospective', 'Pre-cut'], horizontal=True)
confirmed = False
if mode == 'Pre-cut':
    st.warning('HPR records describe completed operations. This dataset does not prove which measurements were available before cutting. ProductKey, StartPos, log position, grade and finished-log dimensions are disabled. DBH may itself have been derived during processing.')
    confirmed = st.checkbox('I confirm DBH and species were recorded and available before the decision in this operation.')
else:
    st.info('Retrospective measurement/process estimation. Finished-log dimensions can be used; results are not claims of pre-cut predictive performance.')
options = [x for x in eligible_features(target, mode) if c['logs'][x].notna().any()]
default = [x for x in (['DBH', 'SpeciesGroupKey'] if mode == 'Pre-cut' else (['DBH', 'ProductKey', 'ButtDiameter_ob'] if target == 'LogLength' else ['LogLength', 'ButtDiameter_ob', 'TopDiameter_ob', 'DBH'])) if x in options]
features = st.multiselect('Input features', options, default=default)
with st.form('training'):
    algorithms = st.multiselect('Algorithms to train and compare', ALGORITHMS, default=['Random Forest'])
    a, b, z = st.columns(3)
    seed = a.number_input('Random seed', 0, 100000, 42)
    fraction = b.slider('Held-out tree fraction', .15, .4, .25, .05)
    folds = z.slider('Training CV folds', 2, 8, 5)
    temporal = st.checkbox('Chronological tree holdout and forward-time CV')
    a, b = st.columns(2)
    depth = a.slider('Maximum tree depth', 2, 20, 8)
    estimators = b.slider('Random forest trees', 50, 400, 150, 50)
    alpha = a.number_input('Ridge alpha', .01, 1000., 1.)
    learning = b.number_input('Boosting learning rate', .01, 1., .1)
    submitted = st.form_submit_button('Train and evaluate', type='primary')
if submitted:
    if not algorithms:
        st.error('Select at least one algorithm.')
    for algorithm in algorithms:
        config = Config(target, features, algorithm, mode, confirmed, int(seed), fraction, temporal, folds, estimators, depth, alpha, learning)
        try:
            with st.spinner('Fitting training folds and evaluating unseen trees…'):
                result = run_experiment(c['logs'], config)
                metadata = {**result['metadata'], 'dataset_sha256': c['dataset'].fingerprint, 'scope_signature': c['signature'], 'source': c['source'], 'filters': c['filters'], 'date_filter': c['dates'], 'recovery': c['dataset'].report['recovered']}
                result['metadata'] = save_experiment(metadata, ROOT / 'experiments')
                st.session_state.active_experiment = result
                st.session_state.pop('shap_result', None)
            st.success(algorithm + ' trained; experiment metadata saved locally.')
        except (ValueError, OSError) as exc:
            st.error(f'{algorithm}: {exc}')
if 'active_experiment' in st.session_state:
    result = st.session_state.active_experiment
    st.markdown('#### Active trained model: ' + result['config'].algorithm + ' → ' + result['config'].target)
    evaluation_view(result)
    st.caption('The last successfully trained model is active for explanations. Saved run details are available in Experiment History.')
