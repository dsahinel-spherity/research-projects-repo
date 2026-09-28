import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from analytics_engine.algorithms import alg
from analytics_engine.algorithms.algorithm import Algorithm
from analytics_engine.data_utils import Logger, df_to_dict


logger = Logger(__name__).get_logger()


config_params = {
    alg.test_size: "test_size",
    alg.random_state: "random_state",
    alg.n_estimators: "n_estimators",
    alg.max_depth: "max_depth",
}


class RandomForestRegressionAlgorithm(Algorithm):

    def __init__(self):

        super().__init__()

        self.X_train = None
        self.X_test = None

        self.y_train = None
        self.y_test = None

        self.y_predict = None

        self.random_state = 42
        self.test_size = 0.20

        self.n_estimators = 200
        self.max_depth = None

        self.model = None


    # =====================================================
    # CONFIG
    # =====================================================

    def _config(
        self,
        config_data: dict,
    ):

        if alg.config in config_data:

            config = config_data[
                alg.config
            ]

            for key, attribute in (
                config_params.items()
            ):

                if key in config:

                    setattr(
                        self,
                        attribute,
                        config[key],
                    )

        return self


    # =====================================================
    # SETUP
    # =====================================================

    def setup(
        self,
        setup_data: dict,
    ):

        self._load_data(
            setup_data=setup_data
        )

        self.pima.fillna(
            0,
            inplace=True,
        )

        self._construct_X()
        self._construct_y()


        if (
            len(
                self.target_columns
            )
            != 1
        ):

            raise ValueError(
                "Random Forest Regression currently "
                "requires exactly one target column."
            )


        if alg.algorithm in setup_data:

            self._config(
                setup_data[
                    alg.algorithm
                ]
            )


        self.X_train, self.X_test, self.y_train, self.y_test = (
            train_test_split(
                self.X,
                self.y,
                test_size=self.test_size,
                random_state=self.random_state,
            )
        )


        return self


    # =====================================================
    # TRAIN
    # =====================================================

    def train(self):

        self.model = (
            RandomForestRegressor(
                n_estimators=
                    int(
                        self.n_estimators
                    ),

                max_depth=
                    self.max_depth,

                random_state=
                    self.random_state,

                n_jobs=-1,
            )
        )


        self.model.fit(
            self.X_train,
            self.y_train.values.ravel(),
        )


        logger.info(
            "Random Forest Regression "
            "has been trained"
        )


        return self


    # =====================================================
    # STANDARD TEST-SET PREDICTION
    # =====================================================

    def predict(self):

        if self.model is None:

            self.train()


        self.y_predict = (
            self.model.predict(
                self.X_test
            )
        )


        target_name = (
            self.target_columns[0]
        )


        prediction_df = pd.DataFrame(
            {
                target_name:
                    self.y_predict
            },
            index=
                self.X_test.index,
        )


        score = (
            self.model.score(
                self.X_test,
                self.y_test.values.ravel(),
            )
        )


        mae = mean_absolute_error(
            self.y_test.values.ravel(),
            self.y_predict,
        )


        feature_importance = {

            column:
                float(value)

            for column, value
            in zip(
                self.X.columns,
                self.model.feature_importances_,
            )
        }


        return {

            "y_predict":
                df_to_dict(
                    prediction_df
                ),

            "score":
                float(score),

            "mae":
                float(mae),

            "feature_importance":
                feature_importance,
        }


    # =====================================================
    # ACTUAL VS PREDICTED
    # =====================================================

    def result(self):

        if self.y_predict is None:

            self.predict()


        target_name = (
            self.target_columns[0]
        )


        result_df = pd.DataFrame(
            {
                "actual":
                    self.y_test[
                        target_name
                    ],

                "predicted":
                    self.y_predict,
            },
            index=
                self.X_test.index,
        )


        return df_to_dict(
            result_df
        )


    # =====================================================
    # USER-SUPPLIED SCENARIO PREDICTION
    # =====================================================

    def predict_input(
        self,
        values: dict,
    ):

        if self.model is None:

            self.train()


        row = pd.DataFrame(
            [
                {
                    column:
                        values.get(
                            column,
                            0,
                        )

                    for column
                    in self.X.columns
                }
            ]
        )


        prediction = (
            self.model.predict(
                row
            )[0]
        )


        return {

            "prediction":
                float(
                    np.asarray(
                        prediction
                    ).squeeze()
                )
        }