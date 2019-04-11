import datetime
import pathlib
import re
from collections import defaultdict

import numpy as np
import pandas as pd

from .helpers.predict import create_submission


# Logits are kewl, never seen anyone doing ensembling this way tbh
def ensemble_submission(
    test_size: int,
    output_size: int,
    predictions_path: pathlib.Path,
    submissions_path: pathlib.Path,
    weighted: bool = True,
    threshold: float = 0.77,
):
    paths = predictions_path.glob("*")
    logits = np.zeros((test_size, output_size))
    models = 0
    for path in paths:
        model_logits = np.genfromtxt(path, delimiter=",")
        accuracy = float(re.search(r"\d\.\d+", str(path)).group())
        if accuracy > threshold:
            if weighted:
                model_logits *= accuracy
            logits += model_logits
            models += 1

    name = "logits_"
    if weighted:
        name += "weighted_"
    name += f"models_{models}_"
    name += str(datetime.datetime.today())
    create_submission(submissions_path / pathlib.Path(name), np.argmax(logits, axis=1))
    return logits
