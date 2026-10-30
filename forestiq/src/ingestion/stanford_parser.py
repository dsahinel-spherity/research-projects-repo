"""Strict, namespace-aware HPR ingestion; explicit salvage of complete stems only."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import re
import numpy as np
import pandas as pd
from lxml import etree as ET
from src.ingestion.schema_validation import validate_schema

NS = 'urn:skogforsk:stanford2010'
N = {'s': NS}
TREE_TEXT = ['StemKey', 'StemNumber', 'SpeciesGroupKey', 'OperatorKey', 'HarvestDate', 'ObjectKey', 'SubObjectKey', 'ProcessingCategory']
TREE_NUM = ['DBH', 'ReferenceDiameter', 'BoomAngle', 'BoomExtension',
            'NoOfSatellitesInUseGNSS', 'AccuracyCoordinateHorizontal',
            'AccuracyCoordinateVertical', 'AccuracyMachineBearing']
TREE_EXTRA = ['FellCutStartTime', 'FellCutEndTime', 'QualityIndicatorGNSS', 'QualityIndicatorBearing']
LOG_TEXT = ['LogKey', 'ProductKey', 'LogscomID', 'CuttingReason', 'CuttingStartTime', 'CuttingEndTime']
DIAMETERS = {f'{part} {bark}': f'{part}Diameter_{bark}' for part in ['Butt', 'Mid', 'Top'] for bark in ['ob', 'ub']}
VOLUMES = {'m3 (price)': 'Volume_price', 'm3sob': 'Volume_ob', 'm3sub': 'Volume_ub'}

@dataclass
class Dataset:
    trees: pd.DataFrame
    logs: pd.DataFrame
    report: dict
    fingerprint: str


def _value(node, tag, deep=True):
    found = node.find(('.//' if deep else '') + 's:' + tag, N)
    if found is None and deep:
        # Recognized vendor extension fields may have a separate namespace.
        matches = node.xpath('.//*[local-name()=$tag]', tag=tag)
        found = matches[0] if len(matches) == 1 else None
    return found.text.strip() if found is not None and found.text and found.text.strip() else None


def _number(value):
    try:
        n = float(value)
        return n if np.isfinite(n) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _coords(node, prefix, inherited_crs=None):
    fields = {prefix + x: np.nan for x in ['Latitude', 'Longitude', 'Altitude', 'Accuracy', 'Satellites', 'VerticalAccuracy']}
    fields[prefix + 'Quality'] = None
    for name in ['CoordinateDate', 'ReceiverPosition', 'CRS']:
        fields[prefix + name] = None
    if node is not None:
        fields[prefix + 'CoordinateDate'] = _value(node, 'CoordinateDate')
        fields[prefix + 'ReceiverPosition'] = node.get('receiverPosition')
        fields[prefix + 'CRS'] = node.get('coordinateReferenceSystem') or inherited_crs or _value(node, 'CoordinateEPSGcode')
        fields.update({prefix + x: _number(_value(node, x)) for x in ['Latitude', 'Longitude', 'Altitude']})
        for name, category in [('Latitude', 'South'), ('Longitude', 'West')]:
            matches = node.xpath('./*[local-name()=$name]', name=name)
            el = matches[0] if len(matches) == 1 else None
            if el is not None and el.get(name.lower() + 'Category') == category:
                fields[prefix + name] = -abs(fields[prefix + name])
        fields[prefix + 'Accuracy'] = _number(node.get('accuracyCoordinate'))
        fields[prefix + 'Satellites'] = _number(node.get('noOfSatellitesInUseGNSS'))
        fields[prefix + 'Quality'] = node.get('qualityIndicatorGNSS')
        fields[prefix + 'VerticalAccuracy'] = _number(_value(node, 'AccuracyCoordinateVertical'))
    return fields


def parse_hpr(source: bytes | str | Path, recover: bool = False) -> Dataset:
    raw = source if isinstance(source, bytes) else Path(source).read_bytes()
    if len(raw) > 200 * 1024 * 1024:
        raise ValueError('File exceeds the 200 MB ingestion limit.')
    # Forbid entity declarations before invoking the XML parser, including UTF-16 inputs.
    if re.search(br'<!\s*(DOCTYPE|ENTITY)', raw.replace(b'\x00', b''), re.I):
        raise ValueError('DTD and entity declarations are not permitted.')
    warnings = []
    payload = raw
    start = raw.find(b'<?xml')
    if start > 0:
        warnings.append(f'Removed {start} bytes of non-XML preamble (legend).')
        payload = raw[start:]
    parser = ET.XMLParser(resolve_entities=False, no_network=True, recover=False, huge_tree=False)
    recovered = False
    try:
        root = ET.fromstring(payload, parser)
    except ET.XMLSyntaxError as exc:
        if not recover:
            raise ValueError(f'Malformed XML: {exc}. Enable explicit complete-tree recovery to salvage intact records.') from exc
        # Conservative recovery only for unprefixed, single-machine HPR files.
        text = payload.decode('utf-8-sig')
        root_tag = re.search(r'<HarvestedProduction\b[^>]*>', text)
        machine_tags = re.findall(r'<Machine\b[^>]*>', text)
        blocks = re.findall(r'<Stem\b[^>]*>.*?</Stem\s*>', text, re.S)
        if not root_tag or len(machine_tags) != 1 or not blocks:
            raise ValueError('Recovery requires a single-machine HPR file with intact, unprefixed Stem blocks.') from exc
        key = re.search(r'<MachineKey>([^<]*)</MachineKey>', text)
        user = re.search(r'<MachineUserID>([^<]*)</MachineUserID>', text)
        metadata = ''.join(m.group(0) for m in [key, user] if m)
        try:
            root = ET.fromstring((root_tag.group(0) + machine_tags[0] + metadata + ''.join(blocks) + '</Machine></HarvestedProduction>').encode(), parser)
        except ET.XMLSyntaxError as recovery_error:
            raise ValueError(f'Complete-tree recovery failed: {recovery_error}. Repair the XML source before ingestion.') from recovery_error
        recovered = True
        orphan_keys = len(re.findall(r'<LogKey\b', text)) - sum(len(re.findall(r'<LogKey\b', b)) for b in blocks)
        warnings.append(f'RECOVERED DATA: original XML is malformed ({exc}). Retained {len(blocks)} complete Stem blocks; excluded {orphan_keys} orphan log-key occurrences and all non-stem content except machine IDs. No missing tree identity was inferred.')
    if root.tag != '{' + NS + '}HarvestedProduction':
        raise ValueError('Expected HarvestedProduction in urn:skogforsk:stanford2010.')
    units = {k: root.get(k) for k in ['diameterUnit', 'lengthUnit', 'volumeUnit']}
    for key, supported in [('diameterUnit', 'mm'), ('lengthUnit', 'cm'), ('volumeUnit', 'm3')]:
        if units[key] != supported:
            raise ValueError(f'{key} must explicitly be {supported}; received {units[key]!r}. Unit conversion is not guessed.')
    trees, logs = [], []
    machine_metadata, product_definitions = [], []
    invalid_numeric = []
    def num(node, tag):
        value = _value(node, tag)
        out = _number(value)
        if value is not None and pd.isna(out):
            invalid_numeric.append({'field': tag, 'value': value})
        return out
    for machine in root.findall('s:Machine', N):
        machine_id = _value(machine, 'MachineKey', False) or _value(machine, 'MachineUserID', False)
        machine_metadata.append({name: _value(machine, name, False) for name in ['MachineKey', 'MachineUserID', 'MachineBaseManufacturer', 'MachineBaseModel']})
        products = {}
        for definition in machine.findall('s:ProductDefinition', N):
            product = {name: _value(definition, name) for name in ['ProductKey', 'ProductName', 'ProductUserID']}
            product['MachineID'] = machine_id
            product_definitions.append(product)
            products[product['ProductKey']] = product
        for stem in machine.findall('.//s:Stem', N):
            row = {tag: _value(stem, tag, False) for tag in TREE_TEXT}
            row.update({tag: num(stem, tag) for tag in TREE_NUM})
            row.update({tag: _value(stem, tag) for tag in TREE_EXTRA})
            row['StemGrade'] = _value(stem, 'GradeValue')
            row['StemGradeProfile'] = ' | '.join(f'{e.get("gradeStartPosition", "?")}:{e.text}' for e in stem.findall('.//s:StemGrade/s:GradeValue', N))
            row['MachineID'] = machine_id
            row['MachineUserID'] = _value(machine, 'MachineUserID', False)
            row['TreeID'] = f'{machine_id}::{row["StemKey"]}' if machine_id and row['StemKey'] else None
            row['StemBunchKey'] = _value(stem, 'StemBunchKey')
            for prefix, markers in [('Base', ['Base machine position', 'Base machine']), ('Crane', ['Crane tip position when felling the tree', 'Attachment center'])]:
                matches = [e for e in stem.findall('s:StemCoordinates', N) if e.get('receiverPosition') in markers]
                coord = matches[0] if len(matches) == 1 else None
                if len(matches) > 1:
                    warnings.append(f'Ambiguous {prefix} position for tree {row["StemKey"]}: left missing.')
                row.update(_coords(coord, prefix, root.get('coordinateReferenceSystem')))
            trees.append(row)
            for log in stem.findall('.//s:Log', N):
                record = {tag: _value(log, tag) for tag in LOG_TEXT}
                record['ProductName'] = products.get(record['ProductKey'], {}).get('ProductName')
                record['ProductUserID'] = products.get(record['ProductKey'], {}).get('ProductUserID')
                record.update({'StemKey': row['StemKey'], 'MachineID': machine_id, 'TreeID': row['TreeID']})
                record['StartPos'] = num(log, 'StartPos')
                measurements = [e for e in log.findall('.//s:LogMeasurement', N) if e.get('logMeasurementCategory') in (None, 'Machine')]
                record['LogLength'] = num(measurements[0], 'LogLength') if len(measurements) == 1 else np.nan
                if len(measurements) > 1:
                    warnings.append(f'Ambiguous machine log length for tree {row["StemKey"]}, log {record["LogKey"]}: left missing.')
                record.update({col: np.nan for col in [*DIAMETERS.values(), *VOLUMES.values()]})
                for tag, attr, mapping in [('LogDiameter', 'logDiameterCategory', DIAMETERS), ('LogVolume', 'logVolumeCategory', VOLUMES)]:
                    for category, col in mapping.items():
                        candidates = log.findall('.//s:' + tag, N)
                        matches = [e for e in candidates if e.get(attr) == category and (e.get('logMeasurementCategory') in (None, 'Machine')) and (e.getparent().get('logMeasurementCategory') in (None, 'Machine'))]
                        if len(matches) == 1:
                            record[col] = _number(matches[0].text)
                            if matches[0].text and pd.isna(record[col]):
                                invalid_numeric.append({'field': col, 'value': matches[0].text})
                        elif len(matches) > 1:
                            warnings.append(f'Ambiguous {col} for tree {row["StemKey"]}, log {record["LogKey"]}: left missing.')
                positions = log.xpath('.//*[local-name()="LogCoordinates"]')
                coord = positions[0] if len(positions) == 1 else None
                if len(positions) > 1:
                    warnings.append(f'Ambiguous drop-off position for tree {row["StemKey"]}, log {record["LogKey"]}: left missing.')
                record.update(_coords(coord, 'Log', root.get('coordinateReferenceSystem')))
                logs.append(record)
    if not trees:
        raise ValueError('No complete tree records were found.')
    log_columns = list(dict.fromkeys(LOG_TEXT + ['ProductName', 'ProductUserID', 'StemKey', 'MachineID', 'TreeID', 'StartPos', 'LogLength'] + list(DIAMETERS.values()) + list(VOLUMES.values()) + list(_coords(None, 'Log'))))
    tree_df, log_df = pd.DataFrame(trees), pd.DataFrame(logs, columns=log_columns)
    for col in ['StartPos', 'LogLength', *DIAMETERS.values(), *VOLUMES.values(), 'LogLatitude', 'LogLongitude', 'LogAltitude', 'LogAccuracy', 'LogSatellites', 'LogVerticalAccuracy']:
        log_df[col] = pd.to_numeric(log_df[col], errors='coerce')
    bad_dates = 0
    for frame, cols in [(tree_df, ['HarvestDate', 'FellCutStartTime', 'FellCutEndTime', 'CraneCoordinateDate', 'BaseCoordinateDate']), (log_df, ['CuttingStartTime', 'CuttingEndTime', 'LogCoordinateDate'])]:
        for col in cols:
            parsed = pd.to_datetime(frame[col], errors='coerce', utc=True, format='mixed')
            bad_dates += int((frame[col].notna() & parsed.isna()).sum())
            frame[col] = parsed
    duplicate_trees = int(tree_df.duplicated(['MachineID', 'StemKey'], keep=False).sum())
    duplicate_logs = int(log_df.duplicated(['MachineID', 'StemKey', 'LogKey'], keep=False).sum())
    missing_ids = int(tree_df[['MachineID', 'StemKey']].isna().any(axis=1).sum() + log_df['LogKey'].isna().sum())
    report = {'trees': len(tree_df), 'logs': len(log_df), 'recovered': recovered, 'warnings': warnings,
              'units': units, 'duplicate_tree_rows': duplicate_trees, 'duplicate_log_rows': duplicate_logs,
              'missing_id_rows': missing_ids, 'invalid_numeric_values': invalid_numeric, 'invalid_timestamps': bad_dates,
              'analytics_safe': not (duplicate_trees or duplicate_logs or missing_ids),
              'schema': validate_schema(root, recovered), 'machines': machine_metadata, 'products': product_definitions,
              'processing_categories': tree_df.ProcessingCategory.fillna('Not recorded').value_counts().to_dict()}
    return Dataset(tree_df, log_df, report, hashlib.sha256(raw).hexdigest())
