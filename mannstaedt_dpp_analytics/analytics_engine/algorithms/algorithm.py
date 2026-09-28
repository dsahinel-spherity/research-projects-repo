"""
algorithm.py
============

Base `Algorithm` class. Every concrete algorithm (Cluster, LinearRegression,
LogisticRegression, FeatureSelection, ...) subclasses this to get the same
three things for free:

  1. Three interchangeable ways to load data: from a local file, from a URL,
     or from a database/API query (see `_load_data` below).
  2. A consistent way to split loaded data into included/excluded/target
     columns and build X (features) / y (target) frames from them.
  3. A `setup()` / `train()` / `predict()` lifecycle that every algorithm
     implements, so the engine (see ../engine.py) can call any of them the
     same way regardless of which algorithm it picked.

Adapted from Spherity's original danalyticAPI. The one part that was
project-specific -- `load_data_from_db`, which called a proprietary football
data feed -- has been replaced with a stub. Everything else here is generic
pandas/sklearn plumbing.
"""
from random import choice

import pandas as pd
import requests

from analytics_engine.algorithms import alg
from analytics_engine.data_utils import (
    Logger,
    df_to_dict,
    json_to_df as read_json,
    excel_to_df as read_excel,
    csv_to_df as read_csv,
)


# Define logger.
logger = Logger(__name__).get_logger()


# Where to look for local files when a request doesn't specify a path.
# Point this at wherever your data lives, e.g. via an env var.
data_folder = ""


# Define reader.
reader = {
    "json": read_json,
    "xlsx": read_excel,
    "csv": read_csv
}


# Define column parameters.
column_params = {
    alg.included: "included_columns",
    alg.excluded: "excluded_columns",
    alg.target: "target_columns"
}


class Algorithm:
    """
    Algorithm class is the parent class of every ML algorithm in this
    package. The main reason for this structure is to provide the same
    data-loading and column-handling behaviour for every algorithm, instead
    of repeating it in each one.

    Parameters:
        :param pd.DataFrame self.pima: the full loaded dataset
        :param pd.DataFrame self.X: feature columns
        :param pd.DataFrame self.y: target column(s)
        :param list self.included_columns:
        :param list self.excluded_columns:
        :param list self.target_columns:
    """
    def __init__(self):
        self.pima = None
        self.X = None
        self.y = None
        self.included_columns = None
        self.excluded_columns = None
        self.target_columns = None
        self.label = None

    def _construct_columns(self):
        if not isinstance(self.included_columns, list):
            self.included_columns = self.pima.columns.tolist()
        excluded_columns = list()
        if isinstance(self.excluded_columns, list):
            excluded_columns += self.excluded_columns
        if isinstance(self.target_columns, list):
            excluded_columns += self.target_columns
        self.included_columns = list(
            set(self.included_columns).difference(
                set(excluded_columns)
            )
        )
        return self

    def _construct_X(self):
        # Train Data.
        try:
            self.X = self.pima[self.included_columns]
        except Exception as err:
            logger.error(
                f"Data X (train features) could not be created: {err}"
            )
        return self

    def _construct_y(self):
        # Target Data.
        if not isinstance(self.target_columns, list):
            target = choice(self.included_columns)
            logger.warning(
                    f"Target feature could not be found, hence "
                    f"following feature has been chosen as "
                    f"target feature randomly: {target}"
            )
            self.target_columns = [target]
            self._construct_columns()
            self._construct_X()
        try:
            self.y = self.pima[self.target_columns]
        except KeyError as err:
            logger.error(
                f"Data y (train target) could not be defined: {err}"
            )
        return self

    def _load_features(self, setup_data: dict):
        if alg.features in setup_data:
            for key, value in column_params.items():
                if key in setup_data[alg.features]:
                    setattr(
                        self,
                        value,
                        setup_data[alg.features][key]
                    )
        else:
            logger.warning(
                "Provided setup_data does not have a features key. "
                "Therefore, all columns in the provided data will "
                "be taken on for the objective algorithm."
            )

    def _load_data(self, setup_data: dict):
        self._load_features(setup_data)
        if alg.data not in setup_data:
            raise ValueError(
                "setup_data must have a 'data' key with one of "
                "'file', 'query' or 'url' in it -- there is no "
                "implicit default data source in this starter kit."
            )
        data = setup_data[alg.data]
        if alg.file in data:
            file = data[alg.file]
            return self.load_data_from_file(**file)
        elif alg.query in data:
            query = data[alg.query]
            return self.load_data_from_db(**query)
        elif alg.url in data:
            url = data[alg.url]
            return self.load_data_from_url(url)
        else:
            raise ValueError(
                "setup_data['data'] must contain one of "
                "'file', 'query' or 'url'."
            )

    def load_data_from_file(self, **file):
        if alg.file_name not in file:
            logger.error(
                "Setup data must have a file_name key in the file load option."
            )
        else:
            extension = file[alg.file_name].split(".")[-1]
            if extension not in reader:
                logger.error(
                    "Provided extension type "
                    f"is not supported: {extension}"
                )
            else:
                if alg.path not in file:
                    file[alg.path] = data_folder
                function = reader[extension]
                data = function(
                    **file
                )
                self.pima = data
        return self._construct_columns()

    def load_data_from_db(self, **kwargs):
        """
        Placeholder. In the original project this called a proprietary
        football-match data feed and has been intentionally removed here --
        it isn't relevant outside that project.

        To wire this up for a new data source (e.g. the Umami analytics API
        this dashboard already talks to via app/services/umami_service.py),
        follow the same shape as load_data_from_file/load_data_from_url:
        fetch/query your data, turn it into a single pandas DataFrame,
        assign it to self.pima, then call self._construct_columns().
        """
        raise NotImplementedError(
            "load_data_from_db is a stub in this starter kit -- implement "
            "it against your own data source, or use load_data_from_file / "
            "load_data_from_url instead (see the docstring above)."
        )

    def load_data_from_url(self, url: str, **kwargs):
        result = requests.get(
            url=url,
            **kwargs
        )
        json_data = result.json()
        data = read_json(
            json_file_itself=json_data,
        )
        self.pima = data
        return self._construct_columns()

    def data(
            self,
            train_data: bool = False,
            target_data: bool = False
    ):
        if train_data:
            if self.X is None:
                self._construct_X()
            data_frame = self.X
        elif target_data:
            if self.y is None:
                self._construct_y()
            data_frame = self.y
        else:
            data_frame = self.pima
        data_frame_json = df_to_dict(
            data_frame=data_frame
        )
        data = {
            "data": data_frame_json,
            "included_columns": self.included_columns,
            "excluded_columns": self.excluded_columns,
            "target_columns": self.target_columns
        }
        return data

    def setup(self, setup_data: dict):
        """
        Setup provides the necessary configuration for the algorithm --
        see alg.py for the shape of setup_data. It also picks between the
        three data-loading options above.

        :param dict setup_data:
        :return:
        """
        # Left as a no-op here so subclasses can call super().setup() and
        # then layer their own config handling on top.
        self._load_data(setup_data=setup_data)
        return self

    def train(self):
        # Left empty for abstract/inheritance definition.
        return self

    def predict(self):
        # Left empty for abstract/inheritance definition.
        return self
