"""
data_utils
==========

Small, dependency-free helpers used across the algorithm classes. In the
original project these lived in a separate `src.utils.Utils` module and a
`src.dataRetrieving.readData` module that weren't part of the reusable
pattern, so they're inlined here to keep this package self-contained.

Nothing in this file is football- or Mannstaedt-specific — it's generic
DataFrame/JSON plumbing and a thin logging wrapper.
"""
import inspect
import logging
from typing import Callable

import pandas as pd


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class Logger:
    """
    Thin wrapper around the standard logging module so every module in the
    package logs consistently without repeating boilerplate.

    Usage:
        logger = Logger(__name__).get_logger()
    """

    def __init__(self, name: str):
        self._logger = logging.getLogger(name)
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
                )
            )
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)

    def get_logger(self) -> logging.Logger:
        return self._logger


# ---------------------------------------------------------------------------
# DataFrame <-> JSON-friendly structures
# ---------------------------------------------------------------------------

def df_to_dict(data_frame: pd.DataFrame) -> dict:
    """Row-indexed dict, e.g. {"0": {"col": value, ...}, ...} — handy when
    the caller needs to look rows up by index."""
    if data_frame is None:
        return {}
    return {
        str(index): row
        for index, row in data_frame.to_dict(orient="index").items()
    }


def df_to_list(data_frame: pd.DataFrame) -> list:
    """List of row dicts, e.g. [{"col": value, ...}, ...] — handy for
    tabular/JSON-array style responses."""
    if data_frame is None:
        return []
    return data_frame.to_dict(orient="records")


# ---------------------------------------------------------------------------
# Reflection
# ---------------------------------------------------------------------------

def get_func_input(func: Callable) -> list:
    """Names of the keyword parameters a bound method accepts, used by the
    engine to validate a caller's `input` dict before invoking it."""
    return [
        name for name in inspect.signature(func).parameters
        if name != "self"
    ]


# ---------------------------------------------------------------------------
# Generic file loaders (replace the original project's DB/feed-specific
# readers — these three are plain pandas, nothing proprietary)
# ---------------------------------------------------------------------------

def json_to_df(json_file_itself: dict = None, path: str = None,
               file_name: str = None, **kwargs) -> pd.DataFrame:
    """Load a JSON payload (already-parsed dict, or a path+file_name) into
    a flat DataFrame via pandas.json_normalize."""
    if json_file_itself is None:
        import json
        import os
        full_path = os.path.join(path or "", file_name)
        with open(full_path, "r") as handle:
            json_file_itself = json.load(handle)
    return pd.json_normalize(json_file_itself)


def excel_to_df(path: str = None, file_name: str = None,
                 sheet_name: str = 0, **kwargs) -> pd.DataFrame:
    import os
    full_path = os.path.join(path or "", file_name)
    return pd.read_excel(full_path, sheet_name=sheet_name)


def csv_to_df(path: str = None, file_name: str = None, **kwargs) -> pd.DataFrame:
    import os
    full_path = os.path.join(path or "", file_name)
    return pd.read_csv(full_path)
