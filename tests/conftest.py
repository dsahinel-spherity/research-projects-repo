from pathlib import Path
import pytest
from src.ingestion.stanford_parser import parse_hpr
from src.preprocessing.feature_engineering import prepare
ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture(scope='session')
def dataset():
    return parse_hpr(next(ROOT.glob('25119*.txt')), recover=True)

@pytest.fixture(scope='session')
def data(dataset):
    return prepare(dataset)[1]
