import json
import pandas as pd
import pydeck as pdk
import streamlit as st
from src.utils.geometry import harvest_area

@st.cache_data(max_entries=16, show_spinner=False)
def footprint(trees, fingerprint):
    return harvest_area(trees, fingerprint)


def area_download(area, name='harvest_area.geojson'):
    if area['features']:
        st.download_button('Download harvest area · GeoJSON', json.dumps(area, indent=2, allow_nan=False), name, 'application/geo+json')
    else:
        st.info(area['properties']['status'])


def harvest_map(trees, logs, area, show_dropoff=False):
    points = trees[['FellingLatitude', 'FellingLongitude', 'TreeID']].dropna(subset=['FellingLatitude', 'FellingLongitude'])
    if points.empty:
        st.info('No verified WGS84 felling points are recorded. Machine positions are not substituted.')
        return
    layers = [pdk.Layer('GeoJsonLayer', area, filled=True, stroked=True, get_fill_color=[20, 10, 154, 25], get_line_color=[20,10,154,220], line_width_min_pixels=2),
              pdk.Layer('ScatterplotLayer', points, get_position='[FellingLongitude, FellingLatitude]', get_fill_color=[20,10,154], get_radius=1.5, radius_min_pixels=3, pickable=True)]
    if show_dropoff:
        drops = logs[['DropOffLatitude','DropOffLongitude','TreeID','LogKey']].dropna(subset=['DropOffLatitude','DropOffLongitude'])
        layers.append(pdk.Layer('ScatterplotLayer', drops, get_position='[DropOffLongitude, DropOffLatitude]', get_fill_color=[229,181,65], get_radius=.8, radius_min_pixels=2, pickable=True))
    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=pdk.ViewState(latitude=float(points.FellingLatitude.mean()), longitude=float(points.FellingLongitude.mean()), zoom=16), map_style='light', tooltip={'text': 'Tree: {TreeID}'}), height=400)
    st.caption('Blue: felling points and estimated footprint. Gold: recorded product positions when enabled. Map tiles need internet; GeoJSON and area calculation work offline.')


def registration_feedback(record):
    """Distinguish successful storage from the source's schema validity."""
    schema = record.get('schema', {})
    status = schema.get('status', 'not_checked')
    st.info(f'Saved {record["tree_count"]} trees and {record["log_count"]} logs. '
            'Available in the dataset selector on the next page load.')
    if status == 'valid':
        st.success(f'Schema check passed — the file follows the StanForD {schema.get("version")} HPR structure.')
        st.caption('This checks the file format, not whether every measurement or location is correct.')
    elif status == 'invalid':
        st.error('Schema check failed — the file does not match the official HPR structure. '
                 'It has been saved, but review the issues before using it.')
        st.caption('Missing fields, unexpected elements or vendor extensions can cause this; the whole dataset is not necessarily unusable.')
    elif status == 'recovered_fragment':
        st.error('Damaged or incomplete file — only recoverable records were registered. '
                 'This is not a complete, schema-valid HPR document. Request a fresh export from the source.')
    else:
        st.warning(f'Schema check unavailable — HPR version {schema.get("version") or "unknown"}. '
                   'The file was saved, but its schema has not been verified. Supported checks: 2.1, 3.2, 3.3, 3.6 and 4.1.')
    if schema.get('errors'):
        with st.expander('Schema issues', expanded=True):
            for issue in schema['errors']:
                st.write(f'Line {issue.get("line", "unknown")}: {issue["message"]}')
            st.caption('Up to 25 schema issues are shown. Correct the source file and register it again.')
