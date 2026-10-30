import plotly.express as px
import streamlit as st
from src.ui import context, chart, number
from src.timber_ui import footprint, harvest_map, area_download
st.title('Timber overview')
st.caption('Production, locations and harvest footprint at a glance.')
c = context()
t, d, v = c['trees'], c['logs'], c['volume']
area = footprint(t, c['dataset'].fingerprint)
for col, label, value in zip(st.columns(4), ['Trees', 'Logs', c['volume_label'], 'Harvest footprint · ha'], [len(t), len(d), d[v].sum(min_count=1), area['properties']['area_ha']]):
    col.metric(label, number(value, 0 if label in ['Trees', 'Logs'] else 2))
st.caption(f'Volume recorded for {d[v].notna().sum()} of {len(d)} logs. Area uses {area["properties"]["located_tree_count"]} recorded felling points in the current scope.')
left, right = st.columns([1.5, 1])
with left:
    st.subheader('Harvest area')
    harvest_map(t, d, area)
    st.caption('Convex hull of felling points, including gaps between trees. An estimated footprint, not a surveyed boundary.')
    area_download(area)
with right:
    st.subheader('Production by product')
    if len(d):
        chart(px.bar(d.groupby('ProductKey', dropna=False)[v].sum(min_count=1).reset_index(), x='ProductKey', y=v, labels={v:'Volume · m³','ProductKey':'Product'}))
        st.write(f'Average log length: **{number(d.LogLength.mean())} cm**')
    else:
        st.info('This file contains felled trees but no processed logs.')
    st.write('Open **Timber Data** for per-log records and exports, or **Analytics** for production trends and measurement checks.')
