import math

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.cluster import silhouette_score
from sklearn.preprocessing import MinMaxScaler

from analytics_engine.algorithms import alg
from analytics_engine.algorithms.algorithm import Algorithm
from analytics_engine.data_utils import (
    Logger,
    df_to_dict,
    df_to_list,
)


# =========================================================
# LOGGER
# =========================================================

logger = Logger(__name__).get_logger()


# =========================================================
# CONFIG MAPPING
# =========================================================

config_params = {
    alg.random_state: "random_state",
    alg.n_cluster: "n_cluster",
}


# =========================================================
# CLUSTER ALGORITHM
# =========================================================

class Cluster(Algorithm):

    def __init__(self):
        super().__init__()

        self.random_state = None
        self.n_cluster = None

        self.kmeans = None

        self.train_flag = False

        # Used by silhouette() to repeat the
        # calculation several times.
        self.repetition = 5


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
    # CREATE NEW CLUSTER INSTANCE
    # =====================================================

    def _new_cluster(self):

        cluster = Cluster()

        cluster.n_cluster = (
            self.n_cluster
        )

        cluster.random_state = (
            self.random_state
        )

        return cluster


    # =====================================================
    # SETUP
    # =====================================================

    def setup(
        self,
        setup_data: dict,
    ):

        # ---------------------------------------------
        # LOAD DATA
        # ---------------------------------------------

        self._load_data(
            setup_data=setup_data
        )


        # ---------------------------------------------
        # HANDLE MISSING VALUES
        # ---------------------------------------------

        self.pima.fillna(
            value=0,
            inplace=True,
        )


        # ---------------------------------------------
        # BUILD FEATURE MATRIX
        # ---------------------------------------------

        self._construct_X()


        # ---------------------------------------------
        # NORMALISE FEATURES
        # ---------------------------------------------

        self.X = pd.DataFrame(

            data=MinMaxScaler().fit_transform(
                X=self.X
            ),

            index=self.X.index,

            columns=self.X.columns,
        )


        # ---------------------------------------------
        # LOAD CONFIG
        # ---------------------------------------------

        if alg.algorithm in setup_data:

            self._config(
                config_data=
                    setup_data[
                        alg.algorithm
                    ]
            )

        else:

            logger.warning(
                f"Setup data has a missing "
                f"crucial key: {alg.algorithm}"
            )


        # ---------------------------------------------
        # DEFAULT CLUSTER COUNT
        # ---------------------------------------------

        if self.n_cluster is None:

            self.n_cluster = 3


        return self


    # =====================================================
    # TRAIN
    # =====================================================

    def train(self):

        self.kmeans = KMeans(
            n_clusters=self.n_cluster,
            random_state=self.random_state,
            n_init="auto",
        )

        try:

            self.kmeans.fit_predict(
                X=self.X
            )

        except Exception as err:

            logger.error(
                "While using KMeans algorithm, "
                f"the following error occurred: {err}"
            )

            self.train_flag = False

        else:

            self.train_flag = True


        return self


    # =====================================================
    # PREDICT
    # =====================================================

    def predict(self):

        if self.kmeans is None:

            self.train()


        if not self.train_flag:

            return None


        labels = pd.DataFrame(

            data=self.kmeans.labels_,

            index=self.X.index,

            columns=[
                "label"
            ],
        )


        centers = pd.DataFrame(

            data=(
                self.kmeans
                .cluster_centers_
                .tolist()
            ),

            index=range(
                self.n_cluster
            ),

            columns=self.X.columns,
        )


        results = {

            "labels":
                df_to_dict(
                    labels
                ),

            "centers":
                df_to_dict(
                    centers
                ),

            "inertia":
                float(
                    self.kmeans.inertia_
                ),
        }


        return results


    # =====================================================
    # RESULT
    # =====================================================

    def result(self):

        if self.kmeans is None:

            self.train()


        if not self.train_flag:

            return None


        labels = pd.DataFrame(

            data=self.kmeans.labels_,

            index=self.X.index,

            columns=[
                "label"
            ],
        )


        result_df = pd.concat(
            [
                self.pima,
                labels,
            ],
            axis="columns",
        )


        return df_to_dict(
            result_df
        )


    # =====================================================
    # VISUALISE
    # =====================================================

    def visualise(self):

        if self.X is None:

            return None


        if len(
            self.X.columns
        ) < 2:

            logger.error(
                "Cluster visualisation requires "
                "at least 2 features."
            )

            return None


        pca_data = PCA(
            n_components=2,
            random_state=self.random_state,
        ).fit_transform(
            X=self.X
        )


        visualisation_data = (
            pd.DataFrame(
                data=pca_data,
                columns=[
                    "x",
                    "y",
                ],
                index=self.X.index,
            )
        )


        cluster = (
            self._new_cluster()
        )


        cluster.X = (
            visualisation_data
        )


        prediction = (
            cluster.predict()
        )


        if prediction is None:

            return None


        labels_dict = (
            prediction[
                "labels"
            ]
        )


        labels_df = (
            pd.DataFrame
            .from_dict(
                labels_dict,
                orient="index",
            )
        )


        labels_df.index = (
            visualisation_data.index
        )


        visualisation_data = (
            pd.concat(
                [
                    visualisation_data,
                    labels_df,
                ],
                axis="columns",
            )
        )


        prediction[
            "visualisation"
        ] = df_to_dict(
            visualisation_data
        )


        return prediction


    # =====================================================
    # ELBOW METHOD
    # =====================================================

    def elbow(self):

        if self.X is None:

            return None


        if self.n_cluster is None:

            self.n_cluster = 3


        max_n = int(
            self.n_cluster
            * math.sqrt(5)
        )


        max_n = max(
            2,
            max_n,
        )


        # Cannot create more clusters than
        # available samples.
        max_n = min(
            max_n,
            len(self.X) - 1,
        )


        cluster_numbers = range(
            2,
            max_n + 1,
        )


        elbow_results = []


        for n_cluster in (
            cluster_numbers
        ):

            cluster = (
                self._new_cluster()
            )


            # IMPORTANT:
            # use the current cluster count
            # being tested.
            cluster.n_cluster = (
                n_cluster
            )


            cluster.X = self.X


            cluster.train()


            if (
                cluster.kmeans
                is None
            ):

                continue


            elbow_results.append(
                {
                    "n_cluster":
                        n_cluster,

                    "error":
                        float(
                            cluster
                            .kmeans
                            .inertia_
                        ),
                }
            )


        return elbow_results


    # =====================================================
    # SILHOUETTE ANALYSIS
    # =====================================================

    def silhouette(self):

        if self.X is None:

            return None


        if self.n_cluster is None:

            self.n_cluster = 3


        max_n = int(
            self.n_cluster
            * math.sqrt(5)
        )


        max_n = max(
            2,
            max_n,
        )


        # Silhouette requires:
        #
        # 2 <= n_clusters < number of samples
        #
        max_n = min(
            max_n,
            len(self.X) - 1,
        )


        cluster_numbers = range(
            2,
            max_n + 1,
        )


        silhouette_results = []


        for n_cluster in (
            cluster_numbers
        ):

            scores = []


            for repetition in range(
                self.repetition
            ):

                cluster = (
                    self._new_cluster()
                )


                # THIS IS THE IMPORTANT FIX.
                #
                # The original code did not
                # assign the loop's current
                # n_cluster value here.
                #
                # That caused every row to
                # test the same cluster count.
                cluster.n_cluster = (
                    n_cluster
                )


                # Optional small variation
                # between repetitions.
                #
                # If random_state was provided,
                # use a deterministic sequence.
                if (
                    self.random_state
                    is not None
                ):

                    cluster.random_state = (
                        self.random_state
                        + repetition
                    )


                cluster.X = self.X


                cluster.train()


                if (
                    cluster.kmeans
                    is None
                ):

                    continue


                score = (
                    silhouette_score(
                        self.X,
                        cluster
                        .kmeans
                        .labels_,
                    )
                )


                scores.append(
                    float(score)
                )


            if not scores:

                continue


            average_score = (
                sum(scores)
                / len(scores)
            )


            silhouette_results.append(
                {
                    "n_cluster":
                        n_cluster,

                    "silhouette":
                        float(
                            average_score
                        ),
                }
            )


        return silhouette_results