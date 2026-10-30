"""PyDeck presentation of selected-log GeoJSON; display dashes are not route data."""
import math
import pydeck as pdk


def trace_deck(collection, visible, fit_area=False):
    features = collection['features']
    layers, locations = [], []
    for feature in features:
        geometry, props = feature['geometry'], feature['properties']
        kind = props['kind']
        if geometry['type'] == 'Point':
            locations.append(geometry['coordinates'])
        if not visible.get(kind, False):
            continue
        if kind == 'estimated_harvest_area':
            layers.append(pdk.Layer('GeoJsonLayer', {'type':'FeatureCollection','features':[feature]}, id='harvest-footprint', filled=True, stroked=True, get_fill_color=[20,10,154,22], get_line_color=[20,10,154,180], line_width_min_pixels=2))
        elif geometry['type'] == 'Point':
            felling = kind == 'felling'
            label = 'Felling point' if felling else 'Recorded product position'
            tooltip = f"{label}\nLogscomID: {props.get('LogscomID') or 'Not recorded'}\nTree: {props['StemKey']} · log: {props['LogKey']}\nTime: {props.get('timestamp') or 'Not recorded'}"
            layers.append(pdk.Layer('ScatterplotLayer', [{'position': geometry['coordinates'], 'tooltip': tooltip}], id=kind, get_position='position', get_fill_color=[29,145,78] if felling else [30,110,230], get_line_color=[255,255,255], stroked=True, line_width_min_pixels=1, radius_units=pdk.types.String('pixels'), radius_max_pixels=10 if felling else 6, get_radius=10 if felling else 6, pickable=True))
        elif kind == 'straight_line_connection':
            start, end = geometry['coordinates']
            # Separate short paths render actual visible gaps without a custom JS extension.
            dashes = []
            for i in range(12):
                a,b = i/12, (i+.55)/12
                dashes.append({'path': [[start[j]+(end[j]-start[j])*v for j in [0,1]] for v in [a,b]], 'tooltip':'Straight-line connection — not a measured transportation route'})
            layers.append(pdk.Layer('PathLayer', dashes, id='reference-connection', get_path='path', get_color=[100,108,126], get_width=2, width_units=pdk.types.String('pixels'), width_max_pixels=2, pickable=True))
    # Draw footprint, then reference line, then the selected markers.
    order = {'harvest-footprint':0, 'reference-connection':1, 'felling':2, 'product_position':3}
    layers.sort(key=lambda layer: order[layer.id])
    polygons = [f for f in features if f['geometry']['type'] == 'Polygon']
    if fit_area and polygons:
        view = pdk.data_utils.compute_view(polygons[0]['geometry']['coordinates'][0])
        view.zoom = max(1., min(18., float(view.zoom)-1))
    elif locations:
        if len(locations) == 1 or locations[0] == locations[-1]:
            view = pdk.ViewState(longitude=locations[0][0], latitude=max(-85.,min(85.,locations[0][1])), zoom=18)
        else:
            view = pdk.data_utils.compute_view(locations)
            view.zoom = max(1., min(19., float(view.zoom)-1)) if math.isfinite(view.zoom) else 18
    else:
        if not polygons:
            return None
        view = pdk.data_utils.compute_view(polygons[0]['geometry']['coordinates'][0])
        view.zoom = max(1., min(18., float(view.zoom)-1))
    return pdk.Deck(layers=layers, initial_view_state=view, map_style='light', tooltip={'text':'{tooltip}'})
