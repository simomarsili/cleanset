# -*- coding: utf-8 -*-
"""Utility functions."""

import functools
import logging

__all__ = [
    'is_command',
]


def is_command(cmds):
    """Given one command returns its path, or None.
    Given a list of commands returns the first recoverable path, or None.
    """
    try:
        from shutil import which  # python3 only
    except ImportError:
        from distutils.spawn import find_executable as which

    if isinstance(cmds, str):
        return which(cmds)

    for cmd in cmds:
        path = which(cmd)
        if path is not None:
            return path

    return None


def open_tempfile():
    """Open a temporary file."""
    import tempfile
    tempfile = tempfile.NamedTemporaryFile
    kwargs = {'delete': True, 'mode': 'r+'}
    return tempfile(**kwargs)


def timeit(func):
    """Timeit decorator."""

    @functools.wraps(func)
    def timed(*args, **kwargs):
        import time
        ts0 = time.time()
        result = func(*args, **kwargs)
        ts1 = time.time()
        logging.debug('%r: %2.4f secs', func, ts1 - ts0)
        return result

    return timed


def get_version():
    """Get package version."""
    import os
    import json
    path_to_version = os.path.join(os.path.dirname(__file__), 'version.json')
    with open(path_to_version, 'r') as f:
        version_data = json.load(f)
        try:
            return version_data['version']
        except KeyError:
            # no version number in version.json
            raise KeyError('check version file: no version number')
