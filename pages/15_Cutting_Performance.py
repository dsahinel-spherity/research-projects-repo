import json
import pandas as pd
import plotly.express as px
import streamlit as st
from src.ui import context, chart
from src.ml.cutting_performance import evaluate_cutting

st.title('Cutting Performance')
st.caption('Compare completed cutting times with expectations for similar measurements. Longer than expected is a review signal, not proof of a fault or wasted time.')
c=context()
algorithm=st.selectbox('Benchmark model',['Random Forest','Ridge Regression'])
coverage=st.select_slider('Requested empirical range coverage',options=[.8,.9,.95],value=.9)
st.caption('Inputs: length, over-bark butt/top diameters and DBH. Five outer folds separate trees. Within each fold, different trees fit the model and calibrate the range. No cutting-time inputs or tree IDs are predictors.')
if st.button('Evaluate cutting performance',type='primary'):
    st.session_state.pop('cutting_performance_result',None)
    try:
        with st.spinner('Evaluating separate trees…'):
            result=evaluate_cutting(c['logs'],algorithm,coverage)
            result['summary'].update(source_sha256=c['dataset'].fingerprint,scope_signature=c['signature'],source=c['source'])
            st.session_state.cutting_performance_result=(c['signature'],result)
    except ValueError as exc:
        st.info(str(exc))
cached=st.session_state.get('cutting_performance_result')
if cached and cached[0]==c['signature']:
    result=cached[1];s=result['summary'];d=result['records']
    st.caption(f"Last run: {s['algorithm']}, {s['requested_coverage']:.0%} requested coverage. Rerun after changing controls.")
    for col,label,value in zip(st.columns(4),['Model MAE (s)','Median baseline MAE (s)','Observed range coverage','Logs above range'],[f"{s['model']['MAE']:.3f}",f"{s['median_baseline']['MAE']:.3f}",f"{s['observed_coverage']:.1%}",s['review_logs']]):
        col.metric(label,value)
    st.caption(f"{s['rows']} logs / {s['trees']} trees evaluated; {s['excluded_rows']} excluded for missing identity, nonpositive/missing duration or unusable measurements. Partial input gaps are median-imputed within fitting folds.")
    if s['model']['MAE']>=s['median_baseline']['MAE']:
        st.warning('The model does not beat the training-median baseline by MAE. Do not rely on its ranking as an established performance benchmark.')
    st.info('Ranges are empirical, calibrated on separate trees. Requested coverage is not guaranteed; compare it with observed coverage. Related logs and a small sample limit reliability. These are retrospective estimates, not pre-cut forecasts or total harvesting productivity.')
    cols=['TreeID','LogKey','ProductKey','CuttingDuration','ExpectedSeconds','LowerSeconds','UpperSeconds','DifferenceSeconds','ImputedInputCount']
    cols=[x for x in cols if x in d]
    labels={'CuttingDuration':'Actual (s)','ExpectedSeconds':'Expected (s)','LowerSeconds':'Lower range (s)','UpperSeconds':'Upper range (s)','DifferenceSeconds':'Actual minus expected (s)','ImputedInputCount':'Missing input count'}
    st.subheader('Cuts longer than the expected range')
    flagged=d.loc[d.Review,cols].rename(columns=labels)
    if flagged.empty:st.info('No evaluated cuts exceeded their expected range.')
    st.dataframe(flagged,hide_index=True,use_container_width=True)
    st.download_button('Download cutting review table',flagged.to_csv(index=False),'cutting_review.csv','text/csv')
    chart(px.scatter(d,x='ExpectedSeconds',y='CuttingDuration',color='Review',hover_data=['TreeID','LogKey'],labels={'ExpectedSeconds':'Expected cutting time (s)','CuttingDuration':'Actual cutting time (s)'},title='Predictions for trees excluded from model fitting'))
    st.subheader('Product summary')
    if 'ProductKey' in d:
        grouped=d.groupby('ProductKey',dropna=False).agg(Logs=('Review','size'),ReviewLogs=('Review','sum'),MeanActualSeconds=('CuttingDuration','mean'),MeanExpectedSeconds=('ExpectedSeconds','mean'),MeanDifferenceSeconds=('DifferenceSeconds','mean')).reset_index()
        st.dataframe(grouped,hide_index=True)
    st.caption('Group differences are associations; they do not establish operator performance, machine condition or recoverable savings.')
    with st.expander('All evaluated and excluded records'):
        st.dataframe(d[cols].rename(columns=labels),hide_index=True)
        st.dataframe(result['excluded'],hide_index=True)
    st.download_button('Download all cutting estimates',d.to_csv(index=False),'cutting_estimates.csv','text/csv')
    st.download_button('Download evaluation details',json.dumps(s,indent=2,allow_nan=False),'cutting_evaluation.json','application/json')
