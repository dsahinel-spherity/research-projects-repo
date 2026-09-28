import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
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


class RandomForestClassificationAlgorithm(
    Algorithm
):

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
                "Random Forest Classification "
                "requires exactly one target."
            )


        target_name = (
            self.target_columns[0]
        )


        self.y = (
            self.y[
                target_name
            ]
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

                stratify=
                    self.y,
            )
        )


        return self


    # =====================================================
    # TRAIN
    # =====================================================

    def train(self):

        self.model = (
            RandomForestClassifier(

                n_estimators=
                    int(
                        self.n_estimators
                    ),

                max_depth=
                    self.max_depth,

                random_state=
                    self.random_state,

                class_weight=
                    "balanced",

                n_jobs=-1,
            )
        )


        self.model.fit(
            self.X_train,
            self.y_train,
        )


        logger.info(
            "Random Forest Classification "
            "has been trained"
        )


        return self


    # =====================================================
    # TEST-SET PREDICTION
    # =====================================================

    def predict(self):

        if self.model is None:

            self.train()


        self.y_predict = (
            self.model.predict(
                self.X_test
            )
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


        predictions = pd.DataFrame(
            {
                "actual":
                    self.y_test,

                "predicted":
                    self.y_predict,
            }
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

            "accuracy":
                float(accuracy),

            "precision":
                float(precision),

            "recall":
                float(recall),

            "f1":
                float(f1),

            "predictions":
                df_to_dict(
                    predictions
                ),

            "feature_importance":
                feature_importance,
        }


    # =====================================================
    # CONFUSION MATRIX
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
                self.model.classes_,
            columns=
                self.model.classes_,
        )


        return df_to_dict(
            matrix_df
        )


    # =====================================================
    # USER SCENARIO PREDICTION
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


        probabilities = (
            self.model.predict_proba(
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
                self.model.classes_,
                probabilities,
            )
        }


        return {

            "prediction":
                int(prediction),

            "probabilities":
                probability_map,
        }