import pathlib

import numpy as np


def create_submission(path: pathlib.Path, predictions):
    pred_with_id = np.stack([np.arange(len(predictions)), predictions], axis=1)
    np.savetxt(
        fname=path,
        X=pred_with_id,
        fmt="%d",
        delimiter=",",
        header="id,label",
        comments="",
    )
