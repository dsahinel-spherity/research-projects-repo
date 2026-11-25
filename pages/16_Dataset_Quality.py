import json
import streamlit as st
from src.ui import context
from src.datasets.quality_report import build_quality_report

st.title('Dataset Quality Report')
st.caption('A transparent checklist for the current filtered records. A passed check does not certify a dataset or authorize a DPP.')
c=context()
st.subheader('Source validation - entire uploaded file')
schema=c['dataset'].report.get('schema',{})
st.write('Schema status: **'+schema.get('status','unknown')+'**')
if c['dataset'].report.get('recovered'):
    st.warning('Recovered incomplete source: this report covers retained records only. Missing source fragments are not repaired or certified.')
with st.expander('Source validation and preparation exclusions'):
    st.json(c['dataset'].report)
    st.json(c['invalid'])
st.caption('Schema, parsing and preparation exclusions describe the whole source. Tables below describe the current filters. Invalid source values may already have become missing during preparation; inspect the source report for those exclusions. Structural identity errors block analytics in the shared importer.')
r=build_quality_report(c['logs'],c['volume']);s=r['summary']
for col,label,value in zip(st.columns(3),['Logs in scope','Logs with listed issues','Listed issues'],[s['rows'],s['rows_with_issues'],s['issue_count']]):col.metric(label,value)
st.subheader('Completeness')
st.dataframe(r['completeness'],hide_index=True,use_container_width=True)
st.caption('Location and time gaps are reported as availability gaps. Product coordinates are recorded positions, not confirmed delivery events. No universal quality score is assigned.')
st.subheader('Issues to review')
level=st.multiselect('Issue levels',['Error','Review','Gap'],default=['Error','Review','Gap'])
st.dataframe(r['issues'].loc[r['issues'].Level.isin(level)],hide_index=True,use_container_width=True)
st.caption('Error: invalid measurement/identity. Review: a value needs investigation. Gap: unavailable information, which may be legitimate for this file version or machine.')
for key,name in [('completeness','completeness'),('issues','quality_issues')]:
    st.download_button('Download '+name.replace('_',' '),r[key].to_csv(index=False),name+'.csv','text/csv')
report={'source':c['source'],'source_sha256':c['dataset'].fingerprint,'scope_signature':c['signature'],'source_validation':c['dataset'].report,'source_preparation_exclusions':c['invalid'],'scope_summary':s,'measurement_volume_definition':c['volume']}
st.download_button('Download quality summary',json.dumps(report,indent=2,default=str,allow_nan=False),'dataset_quality.json','application/json')
