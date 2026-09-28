"""
feature.py
==========

Feature-selection algorithms, split into an unsupervised branch (PCA,
correlation) and a supervised branch (ANOVA, mutual information, Lasso,
chi-square). Follows the same Algorithm pattern as the other files.

Unchanged in logic from Spherity's original danalyticAPI -- only the
imports were updated, and one unused import (mlxtend's
SequentialFeatureSelector, never actually called in this file) was dropped
so this module doesn't pull in an extra dependency for nothing.
"""
import random

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.decomposition import PCA
from sklearn.feature_selection import (
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from analytics_engine.algorithms import alg
from analytics_engine.algorithms.algorithm import Algorithm
from analytics_engine.data_utils import Logger, df_to_dict, df_to_list


# --------FEATURE SELECTION--------


# Define logger.
logger = Logger(__name__).get_logger()


# Define unsupervised and supervised variables.
unsupervised = 0
supervised = 1
method = {
    unsupervised: "unsupervised",
    supervised: "supervised"
}


# Algorithm classes.
function_method = {
    "data": unsupervised,
    "pca": unsupervised,
    "pca_info": unsupervised,
    "correlate": unsupervised,
    "correlated": unsupervised,
    "correlation": unsupervised,
    "anova_regression": supervised,
    "anova_classifier": supervised,
    "mutual_info_regression": supervised,
    "mutual_info_classifier": supervised,
    "lasso": supervised,
    "chi_square": supervised
}


class FeatureSelection:
    def __init__(self):
        self.supervised = FeatureSelectionSupervised()
        self.unsupervised = FeatureSelectionUnsupervised()

    def setup(self, setup_data: dict):
        try:
            function = setup_data[alg.algorithm][alg.function]
        except KeyError as err:
            logger.error(
                f"Provided setup_data does not have a function key: {err}"
            )
        else:
            try:
                objective_method = function_method[function]
            except KeyError as err:
                logger.error(
                    f"Provided setup_data does not have a valid function key: {err}"
                )
            else:
                objective_algorithm = getattr(
                    self,
                    method[objective_method]
                )
                objective_algorithm.setup(
                    setup_data=setup_data
                )
        return self

    def data(
            self,
            train_data: bool = False,
            target_data: bool = False
    ):
        return self.unsupervised.data(
            train_data=train_data,
            target_data=target_data
        )

    def pca(self, n_components: int):
        return self.unsupervised.pca(
            n_components=n_components
        )

    def pca_info(self):
        return self.unsupervised.pca_info()

    def correlate(self):
        return self.unsupervised.correlate()

    def correlated(self):
        return self.unsupervised.correlated()

    def correlation(self):
        return self.unsupervised.correlation()

    def anova_regression(self):
        return self.supervised.anova_regression()

    def anova_classifier(self):
        return self.supervised.anova_classifier()

    def mutual_info_regression(self):
        return self.supervised.mutual_info_regression()

    def mutual_info_classifier(self):
        return self.supervised.mutual_info_classifier()

    def lasso(self):
        return self.supervised.lasso()

    def chi_square(self):
        return self.supervised.chi_square()


# --------UNSUPERVISED--------


config_params_unsupervised = {
    alg.test_size: "test_size",
    alg.random_state: "random_state",
    alg.threshold: "threshold",
    alg.normalise: "normalise"
}


class FeatureSelectionUnsupervised(Algorithm):
    def __init__(self):
        super().__init__()
        self.normalise = True
        self.random_state = None
        self.test_size = 0.2
        self.threshold = 0.9

    def _config(self, config_data: dict):
        if alg.config in config_data:
            config = config_data[alg.config]
            for key, value in config_params_unsupervised.items():
                if key in config:
                    setattr(self, value, config[key])
        return self

    def normalise_x(self):
        # Normalize data X.
        self.X = pd.DataFrame(
            data=MinMaxScaler().fit_transform(
                X=self.X
            ),
            index=self.X.index,
            columns=self.X.columns
        )
        return self

    def setup(self, setup_data: dict):
        # General data loading.
        self._load_data(setup_data=setup_data)
        # Fill the None values in data.
        self.pima.fillna(
            value=0,
            inplace=True
        )
        # Define config.
        if alg.algorithm in setup_data:
            self._config(config_data=setup_data[alg.algorithm])
        else:
            logger.warning(
                f"Setup data has a missing crucial key: {alg.algorithm}"
            )
        # Define data X from pima.
        self._construct_X()
        # Normalise X.
        if self.normalise:
            self.normalise_x()
        return self

    def pca(self, n_components: int):
        # Apply PCA.
        pca_data = PCA(
            n_components=n_components,
            random_state=self.random_state
        ).fit_transform(
            X=self.X
        )
        columns = ['PC{}'.format(i) for i in range(1, n_components + 1)]
        reduced_data = pd.DataFrame(
            data=pca_data,
            columns=columns,
            index=self.X.index
        )
        return df_to_dict(reduced_data)

    def pca_info(self):
        pcs = PCA()
        pcs.fit_transform(
            X=self.X
        )
        info = {
            'Standard deviation': np.sqrt(pcs.explained_variance_),
            'Proportion of variance': pcs.explained_variance_ratio_,
            'Cumulative proportion': np.cumsum(pcs.explained_variance_ratio_),
            'Variance Explanation Ratio': pcs.explained_variance_ratio_,
            'Cumulative Ratio': np.cumsum(pcs.explained_variance_ratio_)
        }
        columns = ['PC{}'.format(i) for i in range(1, len(self.X.columns) + 1)]
        info_data_frame = pd.DataFrame(
            data=info.values(),
            columns=columns,
            index=info.keys()
        ).transpose()
        info_data_frame.astype(float).round(6)
        return df_to_dict(info_data_frame)

    def correlate(self):
        to_drop = list()
        # Drop the columns with all the same value.
        for column in self.X.columns:
            if self.X[column].max() == self.X[column].min():
                to_drop.append(column)

        # Create correlation matrix.
        corr_matrix = self.X.corr().abs()

        # Select upper triangle of correlation matrix.
        upper_tri_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))

        # Find feature columns with correlation greater than the given threshold.
        to_drop += [
            column for column in upper_tri_matrix.columns
            if any(upper_tri_matrix[column] > self.threshold)
        ]

        return to_drop

    def correlated(self):
        to_drop = self.correlate()
        reduced_df = self.X.drop(
            self.X[to_drop],
            axis="columns"
        )
        return df_to_dict(reduced_df)

    def correlation(self):
        # Create correlation matrix.
        corr_matrix = self.X.corr().abs()
        # Select upper triangle of correlation matrix.
        upper_tri_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
        return df_to_dict(upper_tri_matrix)


# --------SUPERVISED--------


config_params_supervised = {
    alg.test_size: "test_size",
    alg.random_state: "random_state",
    alg.normalise: "normalise"
}


class FeatureSelectionSupervised(Algorithm):
    def __init__(self):
        super().__init__()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.normalise = True
        self.random_state = None
        self.test_size = 0.2

    def _config(self, config_data: dict):
        if alg.config in config_data:
            config = config_data[alg.config]
            for key, value in config_params_supervised.items():
                if key in config:
                    setattr(self, value, config[key])
        return self

    def _split(self):
        # Split dataset into training set and test set
        try:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=self.test_size, random_state=self.random_state
            )
        except Exception as err:
            logger.error(
                "While using train_test_split, "
                f"following error occurred: {err}"
            )
        return self

    def normalise_x(self):
        # Normalize data X.
        self.X = pd.DataFrame(
            data=MinMaxScaler().fit_transform(
                X=self.X
            ),
            index=self.X.index,
            columns=self.X.columns
        )
        return self

    def _adjust_y(self):
        total_targets = len(self.target_columns)
        if total_targets > 1:
            objective_target = random.choice(self.target_columns)
            logger.warning(
                f"Most feature selection algorithms require only "
                f"one target feature, however the provided number of target "
                f"features is {total_targets}. Thus, only one of them "
                f"will be chosen: {objective_target}"
            )
            self.target_columns = [objective_target]

    def _convert_y(self):
        if len(self.target_columns) > 0:
            objective_target = self.target_columns[0]
            try:
                self.y = self.y[objective_target]
            except KeyError:
                logger.error(
                    f"Target data frame does not have the column: {objective_target}"
                )

    def setup(self, setup_data: dict):
        # General data loading.
        self._load_data(setup_data=setup_data)
        # Fill the None values in data.
        self.pima.fillna(
            value=0,
            inplace=True
        )
        # Define config.
        if alg.algorithm in setup_data:
            self._config(config_data=setup_data[alg.algorithm])
        else:
            logger.warning(
                f"Setup data has a missing crucial key: {alg.algorithm}"
            )
        # Define data X from pima.
        self._construct_X()
        # Make sure there is only one target.
        self._adjust_y()
        # Define data y from pima.
        self._construct_y()
        # Make sure target y is a Series.
        self._convert_y()
        # Normalise X.
        if self.normalise:
            self.normalise_x()
        # Split dataset into training set and test set
        self._split()
        return self

    def anova_regression(self):
        result = f_regression(
            X=self.X_train,
            y=self.y_train
        )[1]
        data = {
            "feature": self.X.columns,
            "p-value": result
        }
        anova = pd.DataFrame(
            data=data
        ).sort_values(
            by=["p-value"],
            ascending=False
        )
        return df_to_list(anova)

    def anova_classifier(self):
        result = f_classif(
            X=self.X_train,
            y=self.y_train
        )[1]
        data = {
            "feature": self.X.columns,
            "p-value": result
        }
        anova = pd.DataFrame(
            data=data
        ).sort_values(
            by=["p-value"],
            ascending=False
        )
        return df_to_list(anova)

    def mutual_info_regression(self):
        result = mutual_info_regression(
            X=self.X_train,
            y=self.y_train
        )
        data = {
            "feature": self.X.columns,
            "mutual_info": result
        }
        anova = pd.DataFrame(
            data=data
        ).sort_values(
            by=["mutual_info"],
            ascending=False
        )
        return df_to_list(anova)

    def mutual_info_classifier(self):
        result = mutual_info_classif(
            X=self.X_train,
            y=self.y_train
        )
        data = {
            "feature": self.X.columns,
            "mutual_info": result
        }
        mutual_info = pd.DataFrame(
            data=data
        ).sort_values(
            by=["mutual_info"],
            ascending=False
        )
        return df_to_list(mutual_info)

    def lasso(self):
        pipeline = Pipeline(
            steps=[
                ('scaler', StandardScaler()),
                ('model', Lasso())
            ]
        )
        grid = {
            'model__alpha': np.arange(0.1, 10, 0.1)
        }
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=grid,
            cv=5,
            scoring="neg_mean_squared_error",
            verbose=3
        )
        search.fit(
            X=self.X_train,
            y=self.y_train
        )
        coefficients = search.best_estimator_.named_steps['model'].coef_
        data = {
            "feature": self.X.columns,
            "importance": np.abs(coefficients)
        }
        importance = pd.DataFrame(
            data=data
        ).sort_values(
            by=["importance"],
            ascending=False
        )
        return df_to_list(importance)

    def chi_square(self):
        chi_scores = list()
        p_values = list()
        degrees = list()
        for feature in self.X_train.columns:
            # create contingency table
            cross_tab = pd.crosstab(
                index=self.y_train,
                columns=self.X_train[feature]
            )
            # chi_test
            chi, p_value, dof, temp = chi2_contingency(
                observed=cross_tab
            )
            chi_scores.append(chi)
            p_values.append(p_value)
            degrees.append(dof)
        data = {
            "features": self.X_train.columns,
            "chi_score": chi_scores,
            "p_value": p_values,
            "degree_of_freedom": degrees
        }
        scores = pd.DataFrame(
            data=data
        )
        scores.sort_values(
            by=["chi_score"],
            ascending=False
        )
        return df_to_list(scores)
