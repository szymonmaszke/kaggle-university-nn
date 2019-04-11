import pathlib
import sys

import pandas as pd


def print_verbose(message: str, verbose: bool) -> None:
    if verbose:
        print(message, file=sys.stderr)


def train_data(path: pathlib.Path):
    return (
        pd.read_csv(path / "train_data.csv.zip"),
        pd.read_csv(path / "train_labels.csv")["label"],
    )


def test_data(path: pathlib.Path):
    return pd.read_csv(path / "test_data.csv.zip")
