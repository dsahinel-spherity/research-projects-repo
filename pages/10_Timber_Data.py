import streamlit as st
from src.ui import context, ROOT
from src.datasets.registry import register_timber, list_datasets
from src.timber_ui import footprint, harvest_map, area_download, registration_feedback
st.title('Timber Data')
st.caption('One record per log, from the standing tree to its recorded product position.')
c = context()
d, t = c['logs'], c['trees']
registration_key = (c['dataset'].fingerprint, c['recovery'])
if st.button('Register this timber dataset', type='primary'):
    try:
        with st.spinner('Registering dataset and checking the HPR schema…'):
            record = register_timber(c['raw'], c['source'], ROOT/'data/registry', recover=c['recovery'])
        st.session_state['_timber_registration_result'] = (registration_key, record)
    except (ValueError, OSError) as exc:
        st.session_state.pop('_timber_registration_result', None)
        st.error(f'Dataset could not be registered: {exc}')
result = st.session_state.get('_timber_registration_result')
if result and result[0] == registration_key:
    registration_feedback(result[1])
st.caption('Registration saves the complete source and prepared timber tables locally. The downloads below follow the current filters.')
columns = ['TreeID', 'LogKey', 'ProductKey', 'ProductName', c['volume'], 'FellingLatitude', 'FellingLongitude', 'FellingTime', 'DropOffLatitude', 'DropOffLongitude', 'DropOffTime']
st.dataframe(d[columns], use_container_width=True, hide_index=True)
st.download_button('Download timber logs · CSV', d.to_csv(index=False), 'timber_logs.csv', 'text/csv')
st.caption('Felling: crane-tip / attachment-center position; time uses FellCutStartTime, otherwise the labeled HarvestDate. Drop-off: Product position and its CoordinateDate. This timestamp records the coordinate observation, not an independently verified release event. Missing values stay blank; cutting-end time is not substituted.')
with st.expander('Location and time coverage'):
    st.write({'Trees with felling points': int(t.FellingLatitude.notna().sum()), 'Trees with felling times': int(t.FellingTime.notna().sum()), 'Logs with drop-off points': int(d.DropOffLatitude.notna().sum()), 'Logs with drop-off times': int(d.DropOffTime.notna().sum())})
    st.dataframe(d[['TreeID','LogKey','FellingPointSource','FellingTimeSource','DropOffPointSource','DropOffTimeSource']], use_container_width=True, hide_index=True)
area = footprint(t, c['dataset'].fingerprint)
st.subheader('Felling and drop-off locations')
harvest_map(t, d, area, show_dropoff=True)
area_download(area)
with st.expander('Trees and source validation'):
    st.dataframe(t, use_container_width=True)
    st.download_button('Download trees · CSV', t.to_csv(index=False), 'trees.csv', 'text/csv')
    st.json(c['dataset'].report)
with st.expander('Registered datasets'):
    records, errors = list_datasets(ROOT/'data/registry')
    st.dataframe([{k:r.get(k) for k in ['name','tree_count','log_count','area_ha','preparation_branch','registered_at']} for r in records], use_container_width=True)
    for error in errors:
        st.warning(error)
