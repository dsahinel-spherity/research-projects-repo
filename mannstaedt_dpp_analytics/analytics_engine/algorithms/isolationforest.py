import pandas as pd

from sklearn.ensemble import IsolationForest

from analytics_engine.algorithms import alg
from analytics_engine.algorithms.algorithm import Algorithm
from analytics_engine.data_utils import Logger, df_to_dict


logger = Logger(__name__).get_logger()


config_params = {
    alg.random_state: "random_state",
    alg.n_estimators: "n_estimators",
    alg.contamination: "contamination",
}


class IsolationForestAlgorithm(Algorithm):

    def __init__(self):

        super().__init__()

        self.random_state = 42
        self.n_estimators = 200
        self.contamination = 0.05

        self.model = None
        self.labels = None
        self.scores = None


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


        if alg.algorithm in setup_data:

            self._config(
                setup_data[
                    alg.algorithm
                ]
            )


        return self


    # =====================================================
    # TRAIN
    # =====================================================

    def train(self):

        self.model = IsolationForest(

            n_estimators=int(
                self.n_estimators
            ),

            contamination=float(
                self.contamination
            ),

            random_state=
                self.random_state,

            n_jobs=-1,
        )


        self.model.fit(
            self.X
        )


        logger.info(
            "Isolation Forest has been trained"
        )


        return self


    # =====================================================
    # PREDICT
    # =====================================================

    def predict(self):

        if self.model is None:

            self.train()


        raw_labels = (
            self.model.predict(
                self.X
            )
        )


        self.scores = (
            self.model.decision_function(
                self.X
            )
        )


        # sklearn:
        # -1 = anomaly
        #  1 = normal
        #
        # For dashboard clarity:
        # 1 = anomaly
        # 0 = normal

        self.labels = [
            1 if value == -1 else 0
            for value
            in raw_labels
        ]


        result_df = pd.DataFrame(

            {
                "is_anomaly":
                    self.labels,

                "anomaly_score":
                    self.scores,
            },

            index=
                self.X.index,
        )


        return {

            "results":
                df_to_dict(
                    result_df
                ),

            "anomaly_count":
                int(
                    sum(
                        self.labels
                    )
                ),

            "total_rows":
                int(
                    len(
                        result_df
                    )
                ),
        }


    # =====================================================
    # RESULT WITH ORIGINAL DATA
    # =====================================================

    def result(self):

        prediction = (
            self.predict()
        )


        result_df = (
            pd.DataFrame
            .from_dict(
                prediction[
                    "results"
                ],
                orient="index",
            )
        )


        result_df.index = (
            self.pima.index
        )


        combined = pd.concat(

            [
                self.pima,
                result_df,
            ],

            axis=1,
        )


        return df_to_dict(
            combined
        )