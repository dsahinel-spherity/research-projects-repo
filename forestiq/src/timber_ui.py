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
