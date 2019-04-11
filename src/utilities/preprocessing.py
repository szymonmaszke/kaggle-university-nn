import itertools

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from .decorators import logger


def slice_best_features(data, features, to_keep: int):
    slicer = list(map(str, features.index[:to_keep].sort_values()))
    return data[slicer]


def get_best_features(statistical, random_forest):
    def _sort(data, name: str):
        return pd.DataFrame(
            np.arange(len(data)),
            index=data.sort_values(by=name, ascending=False).index,
            columns=[name],
        )

    return pd.DataFrame(
        (
            _sort(statistical, "anova")
            .join(_sort(statistical, "mutual"))
            .join(_sort(random_forest, "importance"))
        ).sum(axis="columns"),
        columns=["importance"],
    ).sort_values(by="importance")


def create_parameters_combinations(parameters):
    for values in itertools.product(*parameters.values()):
        yield {name: value for name, value in zip(parameters.keys(), values)}


def create_models(model_name: str, model, parameters_space):
    return {
        model_name.format(*parameters.values()): model(**parameters, n_jobs=-1)
        for parameters in create_parameters_combinations(parameters_space)
    }


@logger("csv")
def find_outliers(X, y, random_state=None):
    local_outlier_factors_parameters = {
        "n_neighbors": [5, 20, 50, 100],
        "algorithm": ["auto", "kd_tree", "ball_tree"],
        "contamination": ["auto"],
        "random_state": [random_state],
    }

    local_outlier_factors = create_models(
        "LocalOutlierFactor(n_neighbors={}, algorithm={})",
        LocalOutlierFactor,
        local_outlier_factors_parameters,
    )

    isolation_forest_parameters = {
        "n_estimators": [5, 20, 50, 100, 300, 500],
        "bootstrap": [False, True],
        "behaviour": ["new"],
        "contamination": ["auto"],
        "random_state": [random_state],
    }

    isolation_forests = create_models(
        "IsolationForest(n_estimators={}, bootstrap={})",
        IsolationForest,
        isolation_forest_parameters,
    )

    outlier_detectors = {}
    outlier_detectors.update(local_outlier_factors)
    outlier_detectors.update(isolation_forests)

    predictions = {}
    for outlier_detector_name, outlier_detector in outlier_detectors.items():
        predictions[outlier_detector_name] = outlier_detector.fit_predict(X)

    return pd.DataFrame(predictions)
