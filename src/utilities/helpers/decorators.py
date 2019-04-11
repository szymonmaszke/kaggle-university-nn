import pathlib
import pickle

import pandas as pd


def _save_results(results, file: pathlib.Path, method: str):
    if method == "pickle":
        with open(file, "wb") as f:
            pickle.dump(results, f)
    elif method == "csv":
        results.to_csv(file)
    else:
        raise ValueError("Unknown saving method specified in decorator!")
    return results


def _load_results(file: pathlib.Path, method):
    if method == "pickle":
        with open(file, "rb") as f:
            return pickle.load(f)
    if method == "csv":
        return pd.read_csv(file, index_col=0)
    raise ValueError("Unknown loading method specified in decorator!")
