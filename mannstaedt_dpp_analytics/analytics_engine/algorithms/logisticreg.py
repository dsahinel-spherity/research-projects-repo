import random

import pandas as pd

from sklearn.linear_model import (
    LogisticRegression,
)

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from sklearn.model_selection import (
    train_test_split,
)

from analytics_engine.algorithms import alg

from analytics_engine.algorithms.algorithm import (
    Algorithm,
)

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


class LogisticRegressionAlgorithm(
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
    # TARGET HANDLING
    # =====================================================

    def _adjust_y(self):

        total_targets = len(
            self.target_columns
        )


        if total_targets > 1:

            objective_target = (
                random.choice(
                    self.target_columns
                )
            )


            logger.warning(
                "Logistic Regression requires "
                "one target. "
                f"Using: {objective_target}"
            )


            self.target_columns = [
                objective_target
            ]


    def _convert_y(self):

        if len(
            self.target_columns
        ) > 0:

            objective_target = (
                self.target_columns[0]
            )


            self.y = (
                self.y[
                    objective_target
                ]
            )


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

        self._adjust_y()

        self._construct_y()

        self._convert_y()


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

                stratify=
                    self.y,
            )
        )


        return self


    # =====================================================
    # TRAIN
    # =====================================================

    def train(self):

        self.regression = (
            LogisticRegression(

                solver=
                    "liblinear",

                C=
                    10.0,

                random_state=
                    self.random_state,

                class_weight=
                    "balanced",
            )
        )


        self.regression.fit(
            self.X_train,
            self.y_train,
        )


        logger.info(
            "Logistic Regression has been trained"
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


        accuracy = accuracy_score(
            self.y_test,
            self.y_predict,
        )


        precision = precision_score(
            self.y_test,
            self.y_predict,
            zero_division=0,
        )


        recall = recall_score(
            self.y_test,
            self.y_predict,
            zero_division=0,
        )


        f1 = f1_score(
            self.y_test,
            self.y_predict,
            zero_division=0,
        )


        return {

            "y_predict":
                df_to_dict(
                    prediction_df
                ),

            "score":
                float(accuracy),

            "accuracy":
                float(accuracy),

            "precision":
                float(precision),

            "recall":
                float(recall),

            "f1":
                float(f1),
        }


    # =====================================================
    # ACTUAL VS PREDICTED
    # =====================================================

    def result(self):

        if self.y_predict is None:

            self.predict()


        result_df = pd.DataFrame(
            {
                "actual":
                    self.y_test,

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
    # CONFUSION
    # =====================================================

    def confusion(self):

        if self.y_predict is None:

            self.predict()


        matrix = confusion_matrix(
            self.y_test,
            self.y_predict,
        )


        matrix_df = pd.DataFrame(
            matrix,

            index=
                self.regression.classes_,

            columns=
                self.regression.classes_,
        )


        return df_to_dict(
            matrix_df
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


        prediction = int(
            self.regression.predict(
                row
            )[0]
        )


        probabilities = (
            self.regression.predict_proba(
                row
            )[0]
        )


        probability_map = {

            str(
                class_value
            ):
                float(
                    probability
                )

            for class_value, probability
            in zip(
                self.regression.classes_,
                probabilities,
            )
        }


        return {

            "prediction":
                prediction,

            "probabilities":
                probability_map,
        }