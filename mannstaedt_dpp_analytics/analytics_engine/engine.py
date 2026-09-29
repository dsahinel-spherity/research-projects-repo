from analytics_engine.algorithms import alg

from analytics_engine.algorithms.algorithm import (
    Algorithm,
)

from analytics_engine.algorithms.cluster import (
    Cluster,
)

from analytics_engine.algorithms.feature import (
    FeatureSelection,
)

from analytics_engine.algorithms.linearreg import (
    LinearRegressionAlgorithm,
)

from analytics_engine.algorithms.logisticreg import (
    LogisticRegressionAlgorithm,
)

from analytics_engine.algorithms.randomforestreg import (
    RandomForestRegressionAlgorithm,
)

from analytics_engine.algorithms.randomforestclass import (
    RandomForestClassificationAlgorithm,
)

from analytics_engine.algorithms.isolationforest import (
    IsolationForestAlgorithm,
)

from analytics_engine.data_utils import (
    Logger,
    get_func_input,
)


logger = Logger(
    __name__
).get_logger()


# =========================================================
# ALGORITHM REGISTRY
# =========================================================

functions = {

    alg.Algorithm:
        Algorithm,

    alg.LogisticRegression:
        LogisticRegressionAlgorithm,

    alg.LinearRegression:
        LinearRegressionAlgorithm,

    alg.Cluster:
        Cluster,

    alg.FeatureSelection:
        FeatureSelection,

    alg.RandomForestRegression:
        RandomForestRegressionAlgorithm,

    alg.RandomForestClassification:
        RandomForestClassificationAlgorithm,

    alg.IsolationForest:
        IsolationForestAlgorithm,
}


default_algorithm = (
    alg.Algorithm
)

default_purpose = (
    alg.data
)


class AnalyticEngine:

    def __init__(
        self,
        setup_data: dict = None,
    ):

        self.data = (
            setup_data
        )

        self.algorithm = (
            functions[
                default_algorithm
            ]()
        )

        self.purpose = (
            default_purpose
        )

        self.algorithm_function = (
            getattr(
                self.algorithm,
                self.purpose,
            )
        )

        self.input = {}

        logger.info(
            "Analytic engine is started."
        )


    def _get_algorithm(self):

        algorithm = (
            self.data[
                alg.algorithm
            ][
                alg.name
            ]
        )


        if algorithm not in functions:

            raise ValueError(
                "Requested algorithm "
                f"could not be found: {algorithm}"
            )


        self.algorithm = (
            functions[
                algorithm
            ]()
        )


        return self


    def _get_purpose(self):

        purpose = (
            self.data[
                alg.algorithm
            ][
                alg.function
            ]
        )


        if not hasattr(
            self.algorithm,
            purpose,
        ):

            raise AttributeError(
                f"{self.algorithm} does not "
                f"provide method {purpose}."
            )


        self.purpose = (
            purpose
        )


        return self


    def _get_input(self):

        candidate_input = (
            self.data[
                alg.algorithm
            ].get(
                alg.function_input,
                {},
            )
        )


        input_keys_candidate = set(
            candidate_input.keys()
        )


        input_keys = set(
            get_func_input(
                self.algorithm_function
            )
        )


        if not input_keys_candidate.issubset(
            input_keys
        ):

            raise ValueError(
                "Provided input keys do not "
                "match function parameters. "
                f"Allowed: {input_keys}. "
                f"Received: {input_keys_candidate}."
            )


        self.input = (
            candidate_input
        )


        return self


    def _setup_algorithm(self):

        self.algorithm.setup(
            self.data
        )

        return self


    def _run_algorithm(self):

        self._setup_algorithm()

        self._get_input()


        return self.algorithm_function(
            **self.input
        )


    def handle(self):

        self._get_algorithm()

        self._get_purpose()


        self.algorithm_function = (
            getattr(
                self.algorithm,
                self.purpose,
            )
        )


        return self._run_algorithm()