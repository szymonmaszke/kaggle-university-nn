import pathlib
import typing

import numpy as np
import pandas as pd
import torch

from skorch.callbacks import Callback


class PredictTest(Callback):
    def __init__(self, X_test, monitor: str):
        self.X_test = X_test.values.astype(np.float32)
        self.monitor = monitor

    def on_epoch_end(self, net, **_):
        if net.history[-1, f"{self.monitor}_best"]:
            print("Current best validation, making test prediction")
            net.history.best_validation = net.history[-1, self.monitor]
            net.history.predictions = net.predict(self.X_test)


def _ensemble_dataset(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    split_feature: int,
    how_many: int,
    random_state,
):
    train_root = X_train.iloc[:, :split_feature]
    train_ensemble = X_train.iloc[:, split_feature:]

    test_root = X_test.iloc[:, :split_feature]
    test_ensemble = X_test.iloc[:, split_feature:]
    yield X_train, X_test
    for _ in range(how_many):
        additional_features_count = random_state.randint(len(list(train_ensemble)))
        additional_features = random_state.choice(
            list(train_ensemble), size=additional_features_count
        )

        yield pd.merge(
            train_root,
            train_ensemble[additional_features],
            left_index=True,
            right_index=True,
        ), pd.merge(
            test_root,
            test_ensemble[additional_features],
            left_index=True,
            right_index=True,
        )


def _save_predictions(predictions_path: pathlib.Path, name: str, history):
    print(history.best_validation)
    filename = pathlib.Path(f"{history.best_validation}_{name}.csv")
    print(f"Saving predictions in {predictions_path / filename}")
    np.savetxt(predictions_path / filename, history.predictions, delimiter=",")


def _set_seed(seed: int):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def _create_model(input_dim: int, model_config: typing.Dict, output_dim: int):
    def create_layer(input_layer, output_layer, activation, dropout, batch_norm):
        layers = [
            torch.nn.Linear(input_layer, output_layer),
            activation,
            torch.nn.AlphaDropout(dropout),
        ]
        if batch_norm:
            layers.append(torch.nn.BatchNorm1d(output_layer))
        return torch.nn.Sequential(*layers)

    layers = [input_dim] + model_config["layers"]
    layers = [
        create_layer(
            layers[i - 1],
            layers[i],
            model_config["activation"],
            dropout,
            model_config["batch_norm"],
        )
        for i, dropout in zip(range(1, len(layers)), model_config["dropouts"])
    ] + [torch.nn.Linear(model_config["layers"][-1], output_dim)]
    return (model_config["name"], torch.nn.Sequential(*layers))
