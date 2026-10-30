"""Exact identifier search and GeoJSON for recorded timber locations."""
import copy
import json
import math
import re
import pandas as pd

SEARCH_FIELDS = ['LogscomID', 'StemKey', 'LogKey', 'ProductKey']


def identifier(value):
    return '' if value is None or pd.isna(value) else str(value).strip()


def search_logs(logs, field, query):
    if field not in SEARCH_FIELDS:
        raise ValueError('Unsupported log search field.')
    query = str(query).strip()
    if not query:
        return logs.copy()
    return logs.loc[logs[field].map(identifier).eq(query)].copy()


def trace_id(row):
    # Machine scope avoids collisions in multi-machine HPR documents.
    return json.dumps([identifier(row.get(k)) for k in ['DatasetID','MachineID','StemKey','LogKey']], separators=(',', ':'))


def scalar(value):
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, 'item'):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def point(row, prefix):
    try:
        lon, lat = float(row[prefix+'Longitude']), float(row[prefix+'Latitude'])
    except (KeyError, TypeError, ValueError):
        return None
    if math.isfinite(lon) and math.isfinite(lat) and -180 <= lon <= 180 and -90 <= lat <= 90:
        return [lon, lat]
    return None


def selected_geojson(row, area=None, include_area=True):
    properties = {k: scalar(row.get(k)) for k in ['DatasetID','MachineID','LogscomID','StemKey','LogKey','ProductKey','SpeciesGroupKey','LogLength','Volume_price','Volume_ob','Volume_ub']}
    properties.update({'trace_id': trace_id(row), 'length_unit': 'cm', 'volume_unit': 'm3'})
    features = []
    felling, product = point(row, 'Felling'), point(row, 'DropOff')
    for kind, coordinates, time, source in [
        ('felling', felling, 'FellingTime', 'FellingTimeSource'),
        ('product_position', product, 'DropOffTime', 'DropOffTimeSource')]:
        if coordinates is not None:
            features.append({'type':'Feature', 'id':trace_id(row)+':'+kind,
                             'geometry':{'type':'Point','coordinates':coordinates},
                             'properties':{**properties, 'kind':kind, 'timestamp':scalar(row.get(time)),
                                           'timestamp_source':scalar(row.get(source)),
                                           'position_source':scalar(row.get('FellingPointSource' if kind == 'felling' else 'DropOffPointSource')),
                                           'interpretation': 'Recorded felling position' if kind == 'felling' else 'Recorded product position; not a confirmed drop-off event'}})
    if felling is not None and product is not None and felling != product:
        features.append({'type':'Feature', 'id':trace_id(row)+':connection',
                         'geometry':{'type':'LineString','coordinates':[felling, product]},
                         'properties':{**properties,'kind':'straight_line_connection','interpretation':'Straight-line visual connection, not a measured transportation route'}})
    if include_area and area:
        for feature in area.get('features', []):
            if feature.get('geometry', {}).get('type') == 'Polygon':
                polygon = copy.deepcopy(feature)
                polygon['properties']['kind'] = 'estimated_harvest_area'
                features.append(polygon)
    collection = {'type':'FeatureCollection', 'features':features}
    # Reject NaN/Infinity rather than exporting non-standard JSON tokens.
    json.dumps(collection, allow_nan=False)
    return collection


def containment(coordinates, area):
    """Boundary-inclusive point-in-ring check for the engine's simple local hulls."""
    if coordinates is None or not area.get('features'):
        return 'Not assessable'
    ring = area['features'][0]['geometry']['coordinates'][0]
    x,y = coordinates
    inside = False
    for (ax,ay),(bx,by) in zip(ring, ring[1:]):
        dx,dy = bx-ax,by-ay
        length2 = dx*dx+dy*dy
        u = max(0., min(1., ((x-ax)*dx+(y-ay)*dy)/length2)) if length2 else 0.
        if math.hypot(x-(ax+u*dx), y-(ay+u*dy)) <= 1e-10:
            return 'On boundary'
        if (ay > y) != (by > y) and x < (bx-ax)*(y-ay)/(by-ay)+ax:
            inside = not inside
    return 'Inside' if inside else 'Outside'


def download_name(row):
    name = identifier(row.get('LogscomID')) or f"{identifier(row.get('StemKey'))}_{identifier(row.get('LogKey'))}"
    return 'log_' + re.sub(r'[^A-Za-z0-9_.-]', '_', name)[:100] + '_traceability.geojson'
