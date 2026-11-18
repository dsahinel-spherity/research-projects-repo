import hashlib
import json
import pandas as pd
import streamlit as st
from src.ui import context, number
from src.timber_ui import footprint
from src.utils.traceability import SEARCH_FIELDS, search_logs, identifier, trace_id, point, selected_geojson, containment, download_name
from src.traceability_ui import trace_deck

st.title('Log Search & Traceability')
st.caption('Find an individual log and inspect its recorded felling and product positions.')
c = context()
logs = c['logs']
if logs.empty:
    st.info('No processed logs are recorded in this scope. Broaden the filters or select another dataset.')
    st.stop()
field = st.selectbox('Search by', SEARCH_FIELDS, key='trace_field')
query = st.text_input('Search identifier', placeholder='For example: 105', key='trace_query')
st.caption('Exact identifier match. Leave blank to browse the searchable log dropdown. Sidebar filters apply to the search and harvest footprint.')
missing = int(logs.LogscomID.map(identifier).eq('').sum())
st.caption(f'{missing} of {len(logs)} logs have no LogscomID. Find those using StemKey, LogKey or ProductKey, or browse below.')
matches = search_logs(logs, field, query).reset_index(drop=True)
if matches.empty:
    st.info(f'No log matches {field} = {query.strip()!r} in the current scope. Check the identifier or broaden the sidebar filters.')
    st.stop()
if query.strip() and len(matches) > 1:
    st.warning(f'{len(matches)} logs match this identifier. Select the intended machine, tree and log below; no match is chosen automatically.')
labels = {}
rows = {}
for _, row in matches.iterrows():
    key = trace_id(row)
    labels[key] = f"LogscomID {identifier(row.LogscomID) or 'not recorded'} · tree {row.StemKey} · log {row.LogKey} · product {row.ProductKey} · machine {row.MachineID}"
    rows[key] = row
selection_key = hashlib.sha256((c['signature']+field+query).encode()).hexdigest()[:16]
selected = st.selectbox('Matching log (type to search)', list(rows), index=0 if len(rows)==1 else None, format_func=labels.get, key='trace_selected_'+selection_key, placeholder='Choose a log')
if selected is None:
    st.info('Select one matching log to show its locations and traceability details.')
    st.stop()
row = rows[selected]
area = footprint(c['trees'], c['dataset'].fingerprint)
collection = selected_geojson(row, area)
controls = st.columns(4)
visible = {}
for column, kind, label in zip(controls, ['felling','product_position','straight_line_connection','estimated_harvest_area'], ['Felling points','Product positions','Link between recorded points','Harvest-area polygon']):
    visible[kind] = column.checkbox(label, value=kind != 'straight_line_connection', key='trace_layer_'+kind+('_v2' if kind == 'straight_line_connection' else ''), help='Optional visual link between the two recorded positions. No transportation route was measured.' if kind == 'straight_line_connection' else None)
map_col, info_col = st.columns([1.6,1])
with map_col:
    fit_area = st.checkbox('Zoom to harvest footprint', value=False, key='trace_fit_area')
    deck = trace_deck(collection, visible, fit_area=fit_area)
    if deck is not None:
        st.pydeck_chart(deck, height=480, key='trace_map_'+hashlib.sha256(selected.encode()).hexdigest()[:16])
    else:
        st.info('No valid coordinates or harvest polygon are available for this log. The information panel remains available.')
    st.caption('Green: felling point. Blue: recorded product position, not confirmed drop-off. Dashed line: straight-line visual connection, not a measured transportation route. Map tiles require internet.')
    for prefix, label in [('Felling','felling'),('DropOff','product-position')]:
        if point(row,prefix) is None:
            st.info(f'No valid {label} coordinates are recorded; that marker and any connecting line are omitted.')
    if point(row,'Felling') is not None and point(row,'Felling') == point(row,'DropOff'):
        st.caption('The two recorded positions coincide; the smaller blue marker is drawn inside the green marker. No connection line is needed.')
    st.download_button('Download selected log · GeoJSON', json.dumps(collection,indent=2,allow_nan=False), download_name(row), 'application/geo+json')
    st.caption('The export includes all available selected-log geometries and the current footprint. Layer visibility controls only the map.')
with info_col:
    st.subheader('Selected log')
    st.metric('LogscomID', identifier(row.LogscomID) or 'Not recorded')
    st.metric(c['volume_label'], number(row[c['volume']],4))
    st.metric('Length · cm', number(row.LogLength))
    fields = {'Tree / StemKey':row.StemKey, 'LogKey':row.LogKey, 'MachineID':row.MachineID, 'ProductKey':row.ProductKey, 'Product name':row.ProductName, 'SpeciesGroupKey':row.SpeciesGroupKey,
              'Butt diameter · mm, over bark':row.ButtDiameter_ob, 'Mid diameter · mm, over bark':row.MidDiameter_ob, 'Top diameter · mm, over bark':row.TopDiameter_ob,
              'Felling time · UTC':row.FellingTime, 'Product position time · UTC':row.DropOffTime,
              'Felling coordinates · lat, lon':None if point(row,'Felling') is None else f'{row.FellingLatitude:.8f}, {row.FellingLongitude:.8f}',
              'Product coordinates · lat, lon':None if point(row,'DropOff') is None else f'{row.DropOffLatitude:.8f}, {row.DropOffLongitude:.8f}',
              'Tree GNSS quality':row.QualityIndicatorGNSS, 'Product GNSS quality':row.LogQuality, 'Product accuracy (recorded)':row.LogAccuracy}
    st.dataframe(pd.DataFrame([{'Field':k,'Value':'Not recorded' if pd.isna(v) else str(v)} for k,v in fields.items()]), hide_index=True, use_container_width=True)
props = area['properties']
st.subheader('Estimated harvest footprint')
located_trees = set(c['trees'].loc[c['trees'].FellingLatitude.notna() & c['trees'].FellingLongitude.notna(),'TreeID'])
for col,label,value in zip(st.columns(4), ['Area · ha','Trees with felling points','Logs from located trees','Selected felling point'], [number(props['area_ha'],3),len(located_trees),int(logs.TreeID.isin(located_trees).sum()),containment(point(row,'Felling'),area)]):
    col.metric(label,value)
st.caption(f'The current scope has {len(c["trees"])} trees and {len(logs)} logs. The polygon is a convex-hull estimate, not an official legal boundary. Containment includes boundary points and is largely a consistency check because the footprint is generated from these same felling points.')
if not area['features']:
    st.info(props['status'])
with st.expander('Provenance and complete prepared record'):
    st.write('Felling time source:', row.FellingTimeSource)
    st.write('Product-position time source:', row.DropOffTimeSource)
    st.caption('HarvestDate may be a harvest timestamp rather than a precise felling-start measurement. Product-position time is a coordinate observation; it does not independently confirm physical placement. No base-machine position or cutting-end time is substituted.')
    st.dataframe(pd.DataFrame({'Field':row.index,'Value':['Not recorded' if pd.isna(v) else str(v) for v in row.values]}),hide_index=True,use_container_width=True)
