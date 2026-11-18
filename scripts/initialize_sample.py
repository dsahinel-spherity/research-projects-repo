"""Register the included sample with explicit complete-tree recovery."""
from pathlib import Path
from src.datasets.registry import register_timber


def initialize_sample(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    source = root / '25119_KL_Axertal_Wagenburg_Logscom_erklaert_NEU (2).txt'
    return register_timber(source.read_bytes(), source.name, root / 'data/registry', recover=True)


if __name__ == '__main__':
    record = initialize_sample()
    print(f'Registered sample: {record["tree_count"]} trees, {record["log_count"]} logs.')
    print(f'Schema status: {record["schema"]["status"]}. The original file is incomplete; only intact trees were recovered.')
    print(f'Registry ID: {record["id"]}')
