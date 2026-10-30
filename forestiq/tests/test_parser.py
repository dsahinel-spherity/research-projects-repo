from pathlib import Path
import pytest
from lxml import etree as ET
from src.ingestion.stanford_parser import parse_hpr, NS
from src.preprocessing.feature_engineering import prepare

ROOT = Path(__file__).resolve().parents[1]


def valid_xml():
    # Fixture is an actual complete tree from the supplied file, not simulated production data.
    import re
    text = next(ROOT.glob('25119*.txt')).read_text()
    stem = re.search(r'<Stem>.*?</Stem>', text, re.S).group()
    return f'<HarvestedProduction xmlns="{NS}" diameterUnit="mm" lengthUnit="cm" volumeUnit="m3"><Machine><MachineKey>638566229499530000</MachineKey>{stem}</Machine></HarvestedProduction>'.encode()


def test_real_schema_and_recovery(dataset):
    assert len(dataset.trees) == 56
    assert len(dataset.logs) == 323
    assert dataset.report['recovered']
    assert dataset.report['analytics_safe']
    assert any('2 orphan' in w for w in dataset.report['warnings'])
    assert dataset.logs.LogscomID.isna().sum() == 195
    assert dataset.trees.DBH.notna().all()
    assert dataset.logs.Volume_price.eq(0).any()


def test_strict_rejects_broken_source():
    with pytest.raises(ValueError, match='Malformed XML'):
        parse_hpr(next(ROOT.glob('25119*.txt')))


def test_valid_namespace_and_optional_measurements():
    root = ET.fromstring(valid_xml())
    for e in root.findall('.//{' + NS + '}LogLength'):
        e.getparent().remove(e)
    d = parse_hpr(ET.tostring(root))
    assert not d.report['recovered']
    assert d.logs.LogLength.isna().all()
    assert d.logs.Volume_ob.notna().all()


def test_duplicate_relational_id_blocks_analytics():
    import copy
    root = ET.fromstring(valid_xml())
    machine = root[0]
    machine.append(copy.deepcopy(machine[-1]))
    d = parse_hpr(ET.tostring(root))
    assert d.report['duplicate_tree_rows'] == 2
    with pytest.raises(ValueError, match='Duplicate'):
        prepare(d)


def test_wrong_namespace_and_units():
    with pytest.raises(ValueError, match='Expected HarvestedProduction'):
        parse_hpr(valid_xml().replace(NS.encode(), b'wrong'))
    with pytest.raises(ValueError, match='diameterUnit'):
        parse_hpr(valid_xml().replace(b'diameterUnit="mm"', b'diameterUnit="cm"'))


def test_entities_rejected():
    with pytest.raises(ValueError, match='DTD'):
        parse_hpr(b'<!DOCTYPE x [<!ENTITY e SYSTEM "file:///etc/passwd">]><x>&e;</x>')


def test_multi_machine_tree_ids_are_namespaced():
    import copy
    root = ET.fromstring(valid_xml())
    machine = copy.deepcopy(root[0])
    machine[0].text = 'second-machine'
    root.append(machine)
    d = parse_hpr(ET.tostring(root))
    assert d.report['analytics_safe']
    trees, logs, _ = prepare(d)
    assert len(trees) == 2
    assert logs.TreeID.nunique() == 2
