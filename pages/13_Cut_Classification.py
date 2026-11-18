import pandas as pd
import streamlit as st
from src.ui import context, ROOT
from src.ml.classification import Config, FEATURES, cut_labels, run_experiment
from src.ml.evaluation import save_experiment

st.title('Logistic Regression')
st.caption('Predict manual versus automatic cuts.')
c = context()
st.info('Logistic regression classifies recorded cuts retrospectively. Finished-log measurements may reflect the cutting decision; this is not validated pre-cut forecasting.')
st.caption('Labels: Automatic = Automatic; Other manual = Manual (positive class). Other reasons and missing labels are excluded, not guessed. CuttingReason and identifiers are never predictors.')
y = cut_labels(c['logs'])
a, b, z = st.columns(3)
a.metric('Automatic logs', int((y == 0).sum()))
b.metric('Manual logs', int((y == 1).sum()))
z.metric('Excluded labels', int(y.isna().sum()))
options = [f for f in FEATURES if c['logs'][f].notna().any()]
with st.form('cut_classifier'):
    features = st.multiselect('Classification features', options, default=[f for f in ['LogLength', 'MidDiameter_ob', 'DBH'] if f in options])
    a,b = st.columns(2)
    strength = a.number_input('C (inverse regularization strength)', .001, 1000., 1.)
    balanced = b.checkbox('Balance class weights', help='Weights are calculated from training labels. Weighted-model probabilities are not calibrated event frequencies.')
    seed = a.number_input('Classification seed', 0, 100000, 42)
    fraction = b.slider('Held-out tree fraction', .15, .4, .25, .05)
    folds = a.slider('Classification CV folds', 2, 8, 5)
    submitted = st.form_submit_button('Train cut classifier', type='primary')
if submitted:
    st.session_state.pop('cut_classifier_result', None)
    try:
        with st.spinner('Training logistic regression on separate trees…'):
            result = run_experiment(c['logs'], Config(features, strength, balanced, int(seed), fraction, folds))
            result['metadata'] = save_experiment({**result['metadata'], 'dataset_sha256': c['dataset'].fingerprint, 'scope_signature': c['signature'], 'source': c['source'], 'filters': c['filters'], 'date_filter': c['dates'], 'recovery': c['dataset'].report['recovered']}, ROOT/'experiments')
            st.session_state.cut_classifier_result = result
        st.success('Logistic regression trained; results saved in Experiment History.')
    except (ValueError, OSError) as exc:
        st.error(str(exc))
if 'cut_classifier_result' in st.session_state:
    result = st.session_state.cut_classifier_result
    m = result['metadata']
    st.subheader('Held-out evaluation')
    st.caption('Results below belong to the last successful training run. Manual probability >= 0.50 predicts Manual. Probabilities are model estimates, not calibrated guarantees.')
    st.dataframe(pd.DataFrame({'Logistic regression': m['test'], 'Training-majority baseline': m['baseline']}), use_container_width=True)
    st.caption(f'{m["train_rows"]} training logs across {len(m["train_group_ids"])} trees; {m["test_rows"]} test logs across {len(m["test_group_ids"])} different trees. Precision/recall/F1 refer to Manual; undefined precision or recall is reported as zero.')
    st.subheader('Confusion matrix')
    st.dataframe(pd.DataFrame(m['confusion_matrix'], index=['Actual Automatic', 'Actual Manual'], columns=['Predicted Automatic', 'Predicted Manual']))
    with st.expander('Training-only cross-validation and configuration'):
        st.dataframe(pd.DataFrame(m['cv']).rename_axis('Fold'))
        st.caption('Stratified tree-group folds; preprocessing is fitted separately in each fold. AUC and balanced accuracy are unavailable for single-class validation folds.')
        st.json(m['config'])
    st.subheader('Predictions on held-out logs')
    cols = ['TreeID', 'LogKey', 'CuttingReason', 'Actual', 'Predicted', 'ManualProbability']
    st.dataframe(result['heldout'][cols], use_container_width=True, hide_index=True)
    st.download_button('Download cut predictions', result['heldout'][cols].to_csv(index=False), 'cut_classification_predictions.csv', 'text/csv')
