import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from analytics_engine.algorithms import alg
from analytics_engine.algorithms.algorithm import Algorithm
from analytics_engine.data_utils import (
    Logger,
    df_to_dict,
)


config_params = {
    alg.test_size:
        "test_size",

    alg.random_state:
        "random_state",

    alg.criterion:
        "criterion",
}


logger = Logger(
    __name__
).get_logger()


class LinearRegressionAlgorithm(
    Algorithm
):

    def __init__(self):

        super().__init__()

        self.X_train = None
        self.X_test = None

        self.y_train = None
        self.y_test = None

        self.random_state = 42
        self.test_size = 0.20

        self.criterion = None

        self.y_predict = None

        self.regression = None


    # =====================================================
    # CONFIG
    # =====================================================

    def _config(
        self,
        config_data: dict,
    ):

        if alg.config in config_data:

            config = (
                config_data[
                    alg.config
                ]
            )


            for key, value in (
                config_params.items()
            ):

                if key in config:

                    setattr(
                        self,
                        value,
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
            setup_data=
                setup_data
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
                "Linear Regression currently "
                "requires exactly one target."
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

                test_size=
                    self.test_size,

                random_state=
                    self.random_state,
            )
        )


        return self


    # =====================================================
    # TRAIN
    # =====================================================

    def train(self):

        self.regression = (
            LinearRegression()
        )


        self.regression.fit(
            self.X_train,
            self.y_train,
        )


        logger.info(
            "Linear Regression has been trained"
        )


        return self


    # =====================================================
    # TEST PREDICTIONS
    # =====================================================

    def predict(self):

        if self.regression is None:

            self.train()


        self.y_predict = (
            self.regression.predict(
                self.X_test
            )
        )


        prediction_df = pd.DataFrame(
            data=
                self.y_predict,

            index=
                self.X_test.index,

            columns=
                self.target_columns,
        )


        return {

            "y_predict":
                df_to_dict(
                    prediction_df
                ),

            "score":
                float(
                    self.regression.score(
                        self.X_test,
                        self.y_test,
                    )
                ),
        }


    # =====================================================
    # ACTUAL VS PREDICTED
    # =====================================================

    def result(self):

        if self.y_predict is None:

            self.predict()


        prediction = pd.DataFrame(
            data=
                self.y_predict,

            index=
                self.X_test.index,

            columns=
                self.target_columns,
        )


        result_data = (
            self.y_test.join(

                other=
                    prediction,

                lsuffix=
                    " (true)",

                rsuffix=
                    " (predict)",
            )
        )


        return df_to_dict(
            result_data
        )


    # =====================================================
    # USER SCENARIO
    # =====================================================

    def predict_input(
        self,
        values: dict,
    ):

        if self.regression is None:

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
            self.regression.predict(
                row
            )
        )


        value = (
            np.asarray(
                prediction
            )
            .reshape(-1)[0]
        )


        return {

            "prediction":
                float(value)
        }