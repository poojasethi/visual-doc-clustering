import os
from logging import getLogger
from pathlib import Path
from typing import List

logger = getLogger(__name__)


def existing_file(arg):
    """Return `Path(arg)` but raise a `ValueError` if it does not refer to an
    existing file."""
    path = Path(arg)
    if not path.is_file():
        raise ValueError("arg (= {!r}) is not a file".format(arg))

    return path


def existing_directory(arg):
    """Return `Path(arg)` but raise a `ValueError` if it does not refer to an
    existing directory."""
    path = Path(arg)
    if not path.is_dir():
        raise ValueError("arg (= {!r}) is not a directory".format(arg))

    return path


def walk(directory, exclude=None, followlinks=True, hook=None, **kwargs):
    if exclude is None:
        exclude = set()

    for dir_, _, files in os.walk(str(directory), followlinks=followlinks, **kwargs):
        dir_ = Path(dir_)

        if hook is not None:
            maybe_files = hook(directory, dir_, files)
            if maybe_files is not None:
                files = maybe_files

        for file_ in files:
            if file_ not in exclude:
                yield dir_ / file_


def list_dirnames(path: Path) -> List[str]:
    """
    Returns a list of all the top-level directory names under the given path, excluding all files.
    """
    return [p.name for p in path.iterdir() if p.is_dir()]


def only_file(d, *, empty_ok=False, multiple_ok=False):
    """Return the single file in the given directory. If no files exist, raise a `ValueError`
    (unless `empty_ok`)."""
    files = list(Path(d).iterdir())
    if len(files) == 0:
        if not empty_ok:
            raise ValueError("directory (= {!r}) is empty".format(d))

        return None
    elif len(files) == 1:
        return files[0]
    elif multiple_ok:
        logger.warning(f"Found multiple files while expecting one: {files}")
        return files[0]
    else:
        raise ValueError("directory (= {!r}) has {} files".format(d, len(files)))


def load_file(path):
    path = existing_file(path)
    with path.open("r") as o:
        return o.read().strip()
