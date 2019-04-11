import pathlib
import typing

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import make_scorer

import skorch

from .helpers.generator import _generate_config
from .helpers.train import (PredictTest, _create_model, _ensemble_dataset,
                            _save_predictions, _set_seed)


def generate_configs(
    max_layers: int,
    max_width: int,
    min_width: int,
    how_many: int,
    selu_vs_relu_rate: float = 0.7,
    constant: int = 1,
    seed: int = None,
):
    random_state = np.random.RandomState(seed)
    return [
        _generate_config(
            max_layers, max_width, min_width, random_state, selu_vs_relu_rate, constant
        )
        for _ in range(how_many)
    ]


def ensemble_datasets(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    split_feature: int,
    how_many_models: int,
    how_many_datasets,
    seed: int,
):
    random_state = np.random.RandomState(seed)
    return [
        _ensemble_dataset(
            X_train, X_test, split_feature, how_many_datasets, random_state
        )
        for _ in range(how_many_models)
    ]


def predict_with_models(
    predictions_path: pathlib.Path,
    y: pd.DataFrame,
    X_test: pd.DataFrame,
    models_configs: typing.Sequence[typing.Dict],
    datasets: typing.Sequence,
    output_dim: int,
    seed: int,
):
    _set_seed(seed)
    for model_config, dataset_generator in zip(models_configs, datasets):
        for i, (X_train, X_test) in enumerate(dataset_generator):
            name, model = _create_model(len(list(X_train)), model_config, output_dim)
            name = f"{i}_{name}"
            net = skorch.NeuralNet(
                model,
                criterion=torch.nn.CrossEntropyLoss,
                max_epochs=40,
                optimizer=torch.optim.Adam,
                device="cuda",
                batch_size=64,
                train_split=skorch.dataset.CVSplit(0.2, stratified=True),
                iterator_train__shuffle=True,
                callbacks=[
                    skorch.callbacks.EpochScoring(
                        make_scorer(
                            lambda y_true, y_pred: np.mean(
                                y_true == np.argmax(y_pred, axis=-1)
                            )
                        ),
                        name="validation_accuracy",
                        lower_is_better=False,
                    ),
                    skorch.callbacks.EarlyStopping(
                        monitor="validation_accuracy", lower_is_better=False, patience=8
                    ),
                    ("PredictTest", PredictTest(X_test, monitor="validation_accuracy")),
                    skorch.callbacks.LRScheduler(
                        policy=torch.optim.lr_scheduler.ReduceLROnPlateau,
                        monitor="validation_accuracy",
                        mode="max",
                        factor=0.6,
                        patience=2,
                        verbose=True,
                    ),
                ],
            )
            net.fit(X_train.values.astype(np.float32), y.values)
            _save_predictions(predictions_path, name, net.history)
