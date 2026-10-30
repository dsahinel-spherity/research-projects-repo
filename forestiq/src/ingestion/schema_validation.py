"""Offline validation against the official Skogforsk HPR schemas."""
from functools import lru_cache
from pathlib import Path
from lxml import etree as ET

SUPPORTED_VERSIONS = ('2.1', '3.3', '3.6', '4.1')

@lru_cache(maxsize=4)
def _schema(version):
    suffix = version.replace('.', 'p')
    path = Path(__file__).resolve().parents[2] / 'data/schemas' / version.replace('.', '') / f'HarvestedProduction_V{suffix}.xsd'
    return ET.XMLSchema(ET.parse(str(path), ET.XMLParser(no_network=True, resolve_entities=False)))


def validate_schema(root, recovered=False):
    version = root.get('version')
    if recovered:
        return {'status': 'recovered_fragment', 'version': version, 'errors': [], 'note': 'Recovered records are not a complete schema-valid HPR document.'}
    if version not in SUPPORTED_VERSIONS:
        return {'status': 'not_checked', 'version': version, 'errors': [], 'note': 'Offline XSD validation is bundled for 2.1, 3.3, 3.6 and 4.1. Other versions use structural extraction and explicit unit checks.'}
    schema = _schema(version)
    valid = schema.validate(root)
    return {'status': 'valid' if valid else 'invalid', 'version': version,
            'errors': [{'line': e.line, 'message': e.message} for e in list(schema.error_log)[:25]],
            'note': 'XSD findings are reported separately from relational and measurement checks; vendor extensions may fail the official schema.'}
