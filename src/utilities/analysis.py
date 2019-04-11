import pathlib

import numpy as np
import pandas as pd
from scipy.stats import skew, skewtest
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

from .decorators import logger, sorter


@sorter(by="skew")
@logger("csv")
def skewness(_: pathlib.Path, data: pd.DataFrame):
    statistic, pvalue = skewtest(data)
    return pd.DataFrame({"skew": skew(data), "statistic": statistic, "pvalue": pvalue})


@sorter(by="VIF")
@logger("csv")
def vifs(_: pathlib.Path, data: pd.DataFrame) -> pd.DataFrame:
    vif_data = add_constant(data)
    return pd.DataFrame(
        [
            variance_inflation_factor(vif_data.values, i)
            for i in range(vif_data.shape[1])
        ],
        index=["constant"] + list(data),
        columns=["VIF"],
    )


@sorter(by="mean", ascending=True)
@logger("csv")
def mean(_: pathlib.Path, data: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(data.mean(axis=0), columns=["mean"])


@sorter(by="variance", ascending=True)
@logger("csv")
def variance(_: pathlib.Path, data: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(data.var(axis=0), columns=["variance"])


@sorter(by="anova")
@logger("csv")
def feature_importance(_: pathlib.Path, X: pd.DataFrame, y) -> pd.DataFrame:
    anova, anova_p_vals = f_classif(X, y)
    mutual_info = mutual_info_classif(X, y)
    return pd.DataFrame(
        {
            "anova": anova,
            "pvalue": anova_p_vals,
            "mutual": mutual_info,
            "mix": mutual_info * anova,
        }
    )


@sorter(by="importance")
@logger("csv")
def rf_feature_importance(
    _: pathlib.Path, X: pd.DataFrame, y, random_state
) -> pd.DataFrame:
    classifier = RandomForestClassifier(
        n_estimators=100, n_jobs=-1, random_state=random_state
    )
    classifier.fit(X, y)
    return pd.DataFrame({"importance": classifier.feature_importances_})
