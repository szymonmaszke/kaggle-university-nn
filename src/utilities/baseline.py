import pathlib

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from .decorators import logger


@logger("csv")
def baselines(
    _: pathlib.Path, X, y, cv: int = 5, random_state=None, only_nn: bool = False
):
    classifiers = {
        "LR": LogisticRegression(
            multi_class="ovr", solver="lbfgs", random_state=random_state
        ),
        "RF": RandomForestClassifier(
            n_estimators=100, random_state=random_state, n_jobs=-1
        ),
        "NN": MLPClassifier(
            hidden_layer_sizes=(200, 100),
            early_stopping=True,
            random_state=random_state,
        ),
    }

    return pd.DataFrame(
        {
            name: cross_val_score(
                clf, X, y, cv=KFold(n_splits=cv, random_state=random_state), n_jobs=-1
            )
            for name, clf in classifiers.items()
        }
    )
