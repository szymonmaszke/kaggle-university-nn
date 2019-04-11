import functools
import pathlib
import sys
import typing

from .general import print_verbose
from .helpers.decorators import _load_results, _save_results


def logger(method: str, verbose: bool = True):
    def outer(func: typing.Callable):
        @functools.wraps(func)
        def inner(file: pathlib.Path, *args, **kwargs):
            if not file.is_file():
                print_verbose("Results not stored, calculating...", verbose)
                results = _save_results(func(file, *args, **kwargs), file, method)
                print_verbose(
                    f"Results calculated successfully, stored at: {str(file.absolute)}",
                    verbose,
                )
                return results
            print_verbose("Results stored, retrieving...", verbose)
            return _load_results(file, method)

        return inner

    return outer


def sorter(by: str, ascending: bool = False):
    def outer(func: typing.Callable):
        @functools.wraps(func)
        def inner(*args, **kwargs):
            return func(*args, **kwargs).sort_values(by=by, ascending=ascending)

        return inner

    return outer
