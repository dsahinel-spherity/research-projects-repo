"""Timber-specific preparation branch: one row per log, with event provenance."""
import numpy as np
import pandas as pd
from src.preprocessing.feature_engineering import prepare

BRANCH = 'timber_hpr_v1'


def geographic_pair(frame, prefix, product_position=False):
    valid = frame[prefix+'Latitude'].between(-90, 90) & frame[prefix+'Longitude'].between(-180, 180)
    crs = frame[prefix+'CRS'].astype('string').str.upper().str.replace(' ', '', regex=False)
    valid &= crs.isin(['WGS84', '4326', 'EPSG:4326'])
    if product_position:
        valid &= frame[prefix+'ReceiverPosition'].eq('Product position')
    return valid


def prepare_timber(dataset):
    trees, logs, invalid = prepare(dataset)
    for frame in [trees, logs]:
        frame['HarvestObjectID'] = frame.MachineID.astype('string') + '::' + frame.ObjectKey.astype('string')
        felling = geographic_pair(frame, 'Crane')
        for axis in ['Latitude', 'Longitude', 'Altitude']:
            frame['Felling'+axis] = frame['Crane'+axis].where(felling)
        frame['FellingPointSource'] = frame.CraneReceiverPosition.where(felling, 'Not recorded / unsupported CRS')
        frame['FellingTime'] = frame.FellCutStartTime.fillna(frame.HarvestDate)
        frame['FellingTimeSource'] = np.select([frame.FellCutStartTime.notna(), frame.HarvestDate.notna()], ['FellCutStartTime', 'HarvestDate (harvest timestamp)'], default='Not recorded')
    dropped = geographic_pair(logs, 'Log', product_position=True)
    for axis in ['Latitude', 'Longitude', 'Altitude']:
        logs['DropOff'+axis] = logs['Log'+axis].where(dropped)
    logs['DropOffTime'] = logs.LogCoordinateDate.where(dropped)
    logs['DropOffPointSource'] = np.where(dropped, 'LogCoordinates / Product position', 'Not recorded / unsupported position or CRS')
    logs['DropOffTimeSource'] = np.where(logs.DropOffTime.notna(), 'Product-position CoordinateDate', 'Not recorded')
    # CuttingEndTime is deliberately not substituted for a placement timestamp.
    logs['LogID'] = logs.TreeID.astype('string') + '::' + logs.LogKey.astype('string')
    logs['DatasetID'] = dataset.fingerprint
    logs['PreparationBranch'] = BRANCH
    return trees, logs, invalid
