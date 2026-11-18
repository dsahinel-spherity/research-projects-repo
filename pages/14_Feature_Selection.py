import json
import pandas as pd
import streamlit as st
from src.ui import context, ROOT
from src.ml.feature_selection import Config, TARGETS, CLASS_TARGET, applicable_methods, eligible_features, compare
from src.ml.evaluation import save_experiment

st.title('Feature Selection')
st.caption('Find a smaller set of measurements that predicts almost as well as the full set. This is retrospective analysis of completed cuts.')
c = context()
target = st.selectbox('What do you want to predict?', TARGETS, format_func=lambda t: {'CuttingDuration': 'Saw time / cutting duration (seconds)', 'LogLength': 'Log length (cm)', 'Volume_price': 'Price volume (m³)', 'Volume_ob': 'Over-bark volume (m³)', 'Volume_ub': 'Under-bark volume (m³)'}.get(t, t))
options = [x for x in eligible_features(target) if x in c['logs'] and c['logs'][x].notna().any()]
st.info(f'{len(options)} available numeric candidate inputs. IDs, timestamps and categorical codes are excluded. “All inputs” means all inputs you select below.')
if target == CLASS_TARGET:
    st.caption('Automatic = 0; Other manual = 1. Other reasons and missing labels are excluded. Logistic regression evaluates the selected measurements; higher balanced accuracy is better.')
else:
    st.caption('Ridge regression evaluates every selection using the same settings. Lower mean absolute error (MAE) is better. Chi-square is unavailable for numeric targets; choose manual/automatic classification to use it.')
with st.expander('What the methods do'):
    st.markdown('''- **Correlation filter:** prioritizes inputs related to the target, then removes strongly overlapping inputs.
- **F-test:** ranks individual inputs by linear association for numeric targets, or differences between cut classes (ANOVA).
- **Mutual information:** ranks inputs by potentially nonlinear relationships with the target.
- **Lasso:** shrinks regression weights and removes zero-weight inputs; available for numeric targets.
- **PCA:** combines standardized inputs into components and reports retained variation. It still needs all contributing original inputs.
- **Chi-square:** bins measurements using training data, then scores their relationship with manual/automatic labels. It is classification-only.''')
    st.caption('Scores describe associations, not causes. Feature rankings can change between folds. Selection and imputation are learned independently inside each training fold.')
features = st.multiselect('Candidate measurements', options, default=options, key='selection_features_'+target)
with st.form('feature_selection_training'):
    methods = st.multiselect('Methods to compare', applicable_methods(target), default=applicable_methods(target))
    sizes = st.multiselect('Subset sizes / PCA component counts to try', list(range(1, len(features)+1)), default=list(range(1, len(features)+1)))
    a,b = st.columns(2)
    folds = a.slider('Selection CV folds', 2, 8, 5)
    seed = b.number_input('Selection random seed', 0, 100000, 42)
    correlation = a.slider('Remove inputs with absolute correlation at least', .5, 1., .9, .05)
    alpha = b.number_input('Lasso alpha (standardized target)', .0001, 10., .01, format='%.4f')
    if target == CLASS_TARGET:
        tolerance = a.slider('Allowed balanced-accuracy loss (percentage points)', 0, 20, 2) / 100
    else:
        tolerance = a.slider('Allowed MAE increase above best CV result (%)', 0, 30, 5) / 100
    submitted = st.form_submit_button('Compare feature sets', type='primary')
if submitted:
    st.session_state.pop('feature_selection_result', None)
    try:
        with st.spinner('Comparing feature sets on the same training-tree folds…'):
            result = compare(c['logs'], Config(target, features, methods, sizes, int(seed), folds, correlation, alpha, tolerance))
            result = save_experiment({**result, 'task': 'feature_selection', 'dataset_sha256': c['dataset'].fingerprint, 'scope_signature': c['signature'], 'source': c['source'], 'filters': c['filters'], 'date_filter': c['dates'], 'recovery': c['dataset'].report['recovered']}, ROOT/'experiments')
            st.session_state.feature_selection_result = result
        st.success('Comparison complete. Results saved in Experiment History.')
    except (ValueError, OSError) as exc:
        st.error(str(exc))
r = st.session_state.get('feature_selection_result')
if r:
    st.subheader('Last completed comparison: '+r['config']['target'])
    st.caption('These results belong to the saved settings below. Run the comparison again after changing controls.')
    table = pd.DataFrame(r['comparison'])
    st.dataframe(table.rename(columns={'CV mean': 'CV mean '+r['metric'], 'CV SD': 'CV SD '+r['metric']}), use_container_width=True, hide_index=True)
    choice = r['recommendation']; d = r['details'][choice]
    st.success('Suggested original-input set: '+', '.join(d['selected']))
    st.write(f'{choice}: the smallest final-training set among the tested non-PCA candidates within your CV tolerance. This is not proof of the globally smallest possible set.')
    st.caption('PCA is excluded from this recommendation because components still require the original inputs. Constant or entirely missing inputs are removed using each training fold only; actual sizes can differ from requested sizes.')
    st.subheader('Final check on unseen trees')
    st.dataframe(pd.DataFrame({key: value['test'] for key,value in r['details'].items() if 'test' in value}), use_container_width=True)
    st.caption(f'{r["train_rows"]} training logs / {len(r["train_group_ids"])} trees; {r["test_rows"]} held-out logs / {len(r["test_group_ids"])} different trees. Excluded unlabelled/missing-ID rows: {r["excluded_rows"]}. Recommendation uses training CV only; test results assess it afterwards. Repeated tuning against this test set would weaken that separation.')
    inspect = st.selectbox('Inspect a method and size', list(r['details']), key='selection_inspect_'+r['id'])
    details = r['details'][inspect]
    with st.expander('Selected inputs, scores and fold stability', expanded=True):
        st.write('Selected on all training trees: '+', '.join(details['selected']))
        if details['scores'] is not None:
            st.dataframe(pd.Series(details['scores'], name='Method score').sort_values(ascending=False).to_frame())
            st.caption('Scores use different scales across methods and should not be compared directly. F-test and chi-square scores are exploratory; repeated logs are not independent samples for significance claims.')
        st.dataframe(pd.DataFrame({'Fold': range(1,len(details['fold_selections'])+1), 'Selected': [', '.join(x) for x in details['fold_selections']]}), hide_index=True)
        if details['pca_variance'] is not None:
            var = pd.Series(details['pca_variance'], index=details['selected'])
            st.dataframe(pd.DataFrame({'Explained variance': var, 'Cumulative variance': var.cumsum()}))
            st.dataframe(pd.DataFrame(details['pca_loadings'], index=details['selected'], columns=details['source_columns']))
            st.caption('Variance describes input information retained, not predictive accuracy. Loadings show how each component combines measurements.')
    for failure in r['failures']:
        st.warning('Not evaluated: '+failure)
    with st.expander('Saved configuration'):
        st.json(r['config'])
    st.download_button('Download comparison CSV', table.to_csv(index=False), 'feature_selection_comparison.csv', 'text/csv')
    st.download_button('Download full selection report', json.dumps(r, indent=2, allow_nan=False), 'feature_selection_report.json', 'application/json')
