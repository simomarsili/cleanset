# -*- coding: utf-8 -*-
"""Test module."""
# pylint: disable=redefined-outer-name
import os

import pytest


def tests_dir():
    """Return None if no tests dir."""
    cwd = os.getcwd()
    basename = os.path.basename(cwd)
    if basename == 'tests':
        return cwd
    tdir = os.path.join(cwd, 'tests')
    if os.path.exists(tdir):
        return tdir
    return None


@pytest.fixture()
def data():
    """Dataframe fixture."""
    import pandas
    data_file = 'nfl.csv.bz2'
    source = os.path.join(tests_dir(), data_file)
    df = pandas.read_csv(source)
    yield {'df': df}


prms = [(0., 0.2, (438, 83)), (0.2, 0.1, (820, 82)), (0.5, 0.1, (1868, 73)),
        (1., 0.1, (1869, 72))]


@pytest.mark.parametrize('axis, thr, expected', prms)
def test_cleaner(data, axis, thr, expected):
    """Test Cleaner instance."""
    from cleanset import Cleaner
    df = data['df']
    cleaner = Cleaner(fna=thr, axis=axis)
    assert cleaner.fit_transform(df).shape == expected


@pytest.mark.parametrize('axis, thr, expected', prms)
def test_clean(data, axis, thr, expected):
    """Test clean function."""
    from cleanset import clean
    df = data['df']
    rows, cols = clean(df, fna=thr, axis=axis)
    assert (len(rows), len(cols)) == expected
