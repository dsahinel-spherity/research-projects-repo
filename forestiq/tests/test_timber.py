import copy
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from lxml import etree as ET
from src.ingestion.stanford_parser import parse_hpr, NS
from src.preprocessing.timber import prepare_timber, BRANCH
from src.utils.geometry import harvest_area
from src.datasets.registry import register_timber, list_datasets
from tests.test_parser import valid_xml


def test_real_timber_and_area(dataset):
    trees, logs, _ = prepare_timber(dataset)
    assert logs.LogID.is_unique
    assert len(logs) == 323
    assert logs.FellingTime.notna().sum() == 323
    assert logs.DropOffTime.notna().sum() == 323
    assert (logs.DropOffTime == logs.LogCoordinateDate).all()
    area = harvest_area(trees, dataset.fingerprint)
    assert area['properties']['area_ha'] == pytest.approx(1.24, abs=.01)
    ring = area['features'][0]['geometry']['coordinates'][0]
    assert ring[0] == ring[-1]
    assert 7 < ring[0][0] < 8 and 49 < ring[0][1] < 50
    assert len(harvest_area(trees.iloc[:2])['features']) == 0
    assert json.loads(json.dumps(area, allow_nan=False))['type'] == 'FeatureCollection'


def test_no_machine_or_cutting_time_substitution(dataset):
    d = copy.deepcopy(dataset)
    d.trees.loc[:, ['CraneLatitude','CraneLongitude']] = np.nan
    d.logs.loc[:, 'LogCoordinateDate'] = pd.NaT
    trees, logs, _ = prepare_timber(d)
    assert trees.FellingLatitude.isna().all()
    assert logs.DropOffTime.isna().all()
    assert logs.CuttingEndTime.notna().all()
    assert not harvest_area(trees)['features']


def test_non_geographic_crs_not_exported(dataset):
    d = copy.deepcopy(dataset)
    d.trees['CraneCRS'] = 'EPSG:3857'
    d.logs['LogCRS'] = 'unknown'
    trees, logs, _ = prepare_timber(d)
    assert trees.FellingLatitude.isna().all()
    assert logs.DropOffLongitude.isna().all()


def test_full_hpr_modern_coordinates_and_products():
    root = ET.fromstring(valid_xml())
    root.set('coordinateReferenceSystem', 'WGS84')
    machine = root[0]
    definition = ET.SubElement(machine, '{'+NS+'}ProductDefinition')
    key = root.find('.//{'+NS+'}ProductKey').text
    for tag, value in [('ProductKey', key), ('ProductName','Fixture product')]:
        ET.SubElement(definition, '{'+NS+'}'+tag).text = value
    for el in root.findall('.//{'+NS+'}StemCoordinates'):
        if el.get('receiverPosition').startswith('Crane'):
            el.set('receiverPosition','Attachment center')
        else:
            el.set('receiverPosition','Base machine')
        el.attrib.pop('coordinateReferenceSystem', None)
    # Standard direct LogCoordinates, outside the vendor Extension.
    for log in root.findall('.//{'+NS+'}Log'):
        coord = log.find('.//{'+NS+'}LogCoordinates')
        coord.getparent().remove(coord)
        log.append(coord)
    dataset = parse_hpr(ET.tostring(root))
    trees, logs, _ = prepare_timber(dataset)
    assert trees.FellingLatitude.notna().all()
    assert logs.DropOffLatitude.notna().all()
    assert 'Fixture product' in set(logs.ProductName)
    assert not dataset.report['recovered']


def test_felled_only_hpr():
    root = ET.fromstring(valid_xml())
    stem = root.find('.//{'+NS+'}Stem')
    processed = stem.find('{'+NS+'}SingleTreeProcessedStem')
    processed.tag = '{'+NS+'}SingleTreeFelledStem'
    for log in processed.findall('{'+NS+'}Log'):
        processed.remove(log)
    dataset = parse_hpr(ET.tostring(root))
    trees, logs, _ = prepare_timber(dataset)
    assert len(trees) == 1 and logs.empty
    assert 'DropOffTime' in logs


def test_registry_roundtrip(tmp_path):
    raw = valid_xml()
    record = register_timber(raw, '../../full-file.hpr', tmp_path)
    again = register_timber(raw, 'same-file.hpr', tmp_path)
    records, errors = list_datasets(tmp_path)
    assert record == again and len(records) == 1 and not errors
    assert record['preparation_branch'] == BRANCH
    assert record['name'] == 'full-file.hpr'
    assert (tmp_path / record['id'] / 'source.hpr').read_bytes() == raw
    assert len(pd.read_csv(tmp_path/record['id']/'timber_logs.csv')) == record['log_count']
    assert set(record['artifacts']) <= {p.name for p in (tmp_path/record['id']).iterdir()}


def test_geometry_degenerate_and_separate_operations():
    points = pd.DataFrame({'FellingLatitude': [49.,49.,49.], 'FellingLongitude': [7.,7.,7.]})
    assert not harvest_area(points)['features']
    points.loc[2, 'FellingLongitude'] = 30
    points.loc[1, 'FellingLatitude'] = 55
    assert not harvest_area(points)['features']


def complete_hpr_fixture(version='3.3'):
    """Schema-contract fixture using a real tree, with explicitly test-only definitions."""
    source = next(Path(__file__).resolve().parents[1].glob('25119*.txt')).read_text()
    source = source[source.index('<?xml'):]
    header = source[:source.index('</OperatorDefinition>') + len('</OperatorDefinition>')]
    definitions = '''<SpeciesGroupDefinition><SpeciesGroupModificationDate>2025-12-16T06:38:51+01:00</SpeciesGroupModificationDate><SpeciesGroupUserID>fixture-species</SpeciesGroupUserID><SpeciesGroupName>Fixture species</SpeciesGroupName><DBHHeight>130</DBHHeight><Grades><StartGrade>4</StartGrade><Grade><GradeNumber>4</GradeNumber></Grade></Grades><BarkFunction barkFunctionCategory="None" modificationRestricted="false"/><SpeciesGroupKey>530</SpeciesGroupKey></SpeciesGroupDefinition>
    <ProductDefinition><ProductKey>10353</ProductKey><UnclassifiedProductDefinition><ProductName>Fixture product</ProductName><SpeciesGroupKey>530</SpeciesGroupKey></UnclassifiedProductDefinition></ProductDefinition>
    <ObjectDefinition><ObjectUserID>fixture-object</ObjectUserID><ObjectModificationDate>2025-12-16T06:38:51+01:00</ObjectModificationDate><ForestCertification>None</ForestCertification><ObjectKey>101</ObjectKey><StartDate>2025-12-16T06:38:51+01:00</StartDate></ObjectDefinition>'''
    root = ET.fromstring((header + definitions + re.search(r'<Stem>.*?</Stem>', source, re.S).group() + '</Machine></HarvestedProduction>').encode())
    for el in root.findall('.//{'+NS+'}Extension'):
        el.getparent().remove(el)
    for tag in ['DBHHeight', 'Grades']:
        root.find('.//{'+NS+'}'+tag).set('modificationRestricted', 'false')
    if version == '4.1':
        root.set('version', '4.1')
        root.set('versionDate', '2025-02-05')
        root.set('coordinateReferenceSystem', 'WGS84')
        root.find('.//{'+NS+'}CountryCode').set('countryCodeCategory', 'ISO 3166')
        bark = root.find('.//{'+NS+'}BarkFunction')
        bark.getparent().remove(bark)
        for coord in root.findall('.//{'+NS+'}StemCoordinates'):
            coord.set('receiverPosition', 'Attachment center' if coord.get('receiverPosition').startswith('Crane') else 'Base machine')
            coord.attrib.pop('coordinateReferenceSystem', None)
    return ET.tostring(root)


@pytest.mark.parametrize('version', ['3.3','4.1'])
def test_complete_schema_valid_hpr(version, tmp_path):
    path = tmp_path / 'complete.hpr'
    path.write_bytes(complete_hpr_fixture(version))
    dataset = parse_hpr(path)
    assert dataset.report['schema']['status'] == 'valid', dataset.report['schema']
    trees, logs, _ = prepare_timber(dataset)
    assert len(trees) == 1 and len(logs) > 0
    assert logs.ProductName.eq('Fixture product').all()
    assert logs.DropOffTime.isna().all()  # Vendor events absent in a standard file.
