import re

import numpy as np
import pandas as pd


def load_logits(path, threshold: float):
    paths = path.glob("*")
    models_logits = []
    for path in paths:
        accuracy = float(re.search(r"\d\.\d+", str(path)).group())
        if accuracy > threshold:
            model_logits = accuracy * np.genfromtxt(path, delimiter=",")
            models_logits.append(model_logits.T)
    return np.stack(models_logits, axis=1)


def correlation_of_logit(models_logits):
    return pd.DataFrame(models_logits.T).corr().values


def create_groups(correlation_indices):
    current_group = 0
    group_pointer = np.full_like(correlation_indices, -1, dtype=int)
    visited = np.zeros_like(correlation_indices, dtype=bool)

    def visit_subtrees(index):
        if not visited[index]:
            visited[index] = True
            group_pointer[index] = visit_subtrees(correlation_indices[index])
        elif group_pointer[index] == -1:
            nonlocal current_group
            current_group += 1
            return current_group - 1
        return group_pointer[index]

    for index, _ in enumerate(correlation_indices):
        if not visited[index]:
            visited[index] = True
            group_pointer[index] = visit_subtrees(correlation_indices[index])

    groups = []
    for group in np.unique(group_pointer):
        groups.append(np.nonzero(group_pointer == group)[0])

    return groups


# Inverse of correlation when multiplying
def uniquensemble(path, threshold: float):
    def recursively_mix(logits):
        totally_grouped = 0
        grouped_logits = []
        for logit in logits:
            if logit.shape[0] == 1:
                totally_grouped += 1
                grouped_logits.append(logit)
            else:
                correlations = correlation_of_logit(logit)
                np.fill_diagonal(correlations, 0.0)
                correlation_indices = np.argmax(correlations, axis=1)
                groups = []
                for group in create_groups(correlation_indices):
                    new_group = np.zeros(logit.shape[-1], dtype=np.float64)
                    for model_index in group:
                        new_group += logit[model_index]
                    groups.append(new_group / len(group))
                grouped_logits.append(np.stack(groups, axis=0))

        if totally_grouped == len(logits):
            return np.stack(grouped_logits).squeeze().argmax(axis=0)
        return recursively_mix(grouped_logits)

    loaded_logits = load_logits(path, threshold)
    return recursively_mix(loaded_logits)


def correlation_of_predictions(path, threshold: float = 0.77):
    paths = path.glob("*")
    models_predictions = []
    for path in paths:
        accuracy = float(re.search(r"\d\.\d+", str(path)).group())
        if accuracy > threshold:
            model_logits = np.genfromtxt(path, delimiter=",")
            predictions = np.argmax(model_logits, axis=1)
            models_predictions.append(predictions)
    return pd.DataFrame(np.vstack(models_predictions).T).corr()


# Dumb idea, was tired, w/e
def majority_vote_submission(submissions_path: pathlib.Path):
    predictions = []
    models = []
    for path in submissions_path.glob("*"):
        ensemble_prediction = np.genfromtxt(path, delimiter=",")
        models.append(int(re.search(r"\d+", str(path)).group()))
        predictions.append(ensemble_prediction[1:, 1])
    predictions = np.vstack(predictions)
    predictions = pd.DataFrame(predictions).mode(axis=0).values[0]
    create_submission(
        submissions_path / pathlib.Path(f"majority_{models}"), predictions
    )
