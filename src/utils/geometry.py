"""Convex-hull footprint in a local projection, with WGS84 geodesic area."""
import numpy as np
from pyproj import CRS, Transformer, Geod
from scipy.spatial import ConvexHull, QhullError


def harvest_area(trees, dataset_id=None):
    coords = trees[['FellingLongitude', 'FellingLatitude']].dropna()
    coords = coords[coords.FellingLatitude.between(-90, 90) & coords.FellingLongitude.between(-180, 180)]
    points = coords.drop_duplicates().to_numpy(dtype=float)
    properties = {'method': 'convex_hull_of_recorded_felling_points', 'area_ha': None,
                  'tree_count': len(trees), 'located_tree_count': len(coords), 'unique_points': len(points),
                  'excluded_tree_count': len(trees)-len(coords), 'source_dataset_sha256': dataset_id,
                  'area_method': 'WGS84 ellipsoidal geodesic',
                  'interpretation': 'Estimated footprint, not a surveyed harvest boundary; gaps between trees are included.'}
    collection = {'type': 'FeatureCollection', 'features': [], 'properties': properties}
    if len(points) < 3:
        properties['status'] = 'At least three distinct recorded felling points are required.'
        return collection
    # One local operational footprint. Widely separated operations must be selected separately.
    lon0 = np.rad2deg(np.arctan2(np.sin(np.deg2rad(points[:,0])).mean(), np.cos(np.deg2rad(points[:,0])).mean()))
    lat0 = points[:,1].mean()
    geod = Geod(ellps='WGS84')
    _, _, distance = geod.inv(np.full(len(points),lon0), np.full(len(points),lat0), points[:,0],points[:,1])
    if max(distance) > 100000 or np.ptp(points[:,0]) > 180:
        properties['status'] = 'Points span distant operations or the antimeridian. Select a single harvest object.'
        return collection
    local = CRS.from_proj4(f'+proj=laea +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m')
    transformer = Transformer.from_crs('EPSG:4326', local, always_xy=True)
    xy = np.column_stack(transformer.transform(points[:,0],points[:,1]))
    try:
        hull = ConvexHull(xy)
    except QhullError:
        properties['status'] = 'Felling points are collinear; a polygon cannot be formed.'
        return collection
    ring = points[hull.vertices].tolist()
    area, perimeter = geod.polygon_area_perimeter(*np.array(ring).T)
    if area < 0:
        ring.reverse()
    ring.append(ring[0].copy())
    properties.update({'area_ha': abs(float(area))/10000, 'perimeter_m': float(perimeter), 'status': 'calculated'})
    collection['features'] = [{'type': 'Feature', 'geometry': {'type': 'Polygon', 'coordinates': [ring]}, 'properties': properties.copy()}]
    return collection
