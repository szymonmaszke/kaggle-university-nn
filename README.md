# First Kaggle competition

This part of repository holds my submission to first Kaggle competition.

Folder Structure:
 - `analysis`: data obtained during initial analysis of input data
 - `preprocessing`: data obtained during preprocessing. Final preprocessed data __is
 not__ included and has to be calculated separately
 - `src`: notebooks and other scripts which go through data loading up to creating a
 timestamped submission.

## Source code

Notebooks should be run in the following order:

 1.  `analyse.ipynb` - data analysis (who would of guessed)
 2.  `preprocess.ipynb` - preprocessing and creation of data used during training
 3.  `train.ipynb` - training neural networks used in final submission. This part may
     run up to 8 hours or more, so be careful.
 4.  `predict.ipynb` - ensembling final submission

Some of the files use `caching` in order to speed up computations (e.g. calculation of
Variance Inflation Factor). If data is present (in of the folders `analysis` or
`preprocessing`) it is simply loaded (and eventually sorted). If you wish to turn this
mechanism off (I would advise against it), just delete cached data.

### Utilities

Notebooks merely display results. Core is inside various utilities functions in
`/src/utilities`. Currently
those are undocumented, but should be quite readable (documentation will be updated if
needed) and one should be able to follow (more or less at least).

If you need any clarification see [Contact subsection](#contact).

## Submission

Some of the submissions (including the best one) will be attached via Google Drive if
needed (current size of over `1GB`).

All seeds (especially the one in `train.ipynb` should be set correctly in order to
replicate my results).
To save some time you can check cached outputs of notebooks which display each model and
it's architecture, it's training history and so on.

## Requirements

In order to run this code one requires third party libraries, namely:
- `pytorch` (easy to guess)
- `pandas`
- `sklearn`
- `skorch`
- `statsmodels`

Exact conda versions are provided inside `environment.yml` and using this file is
advised. For more information on environment replication [see here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).


<a name="contact"></a>
## Contact

If you encounter any troubles contact me via e-mail: `szymon.maszke@protonmail.com`
or create an issue in this repository and I will try to respond.
