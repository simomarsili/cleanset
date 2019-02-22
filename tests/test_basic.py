import pytest


def get_tests_dir():
    """Return None is no tests dir."""
    import os
    cwd = os.getcwd()
    basename = os.path.basename(cwd)
    if basename == 'tests':
        return cwd
    else:
        tests_dir = os.path.join(cwd, 'tests')
        if os.path.exists(tests_dir):
            return tests_dir


@pytest.fixture()
def data():
    import os
    import pandas
    data_file = 'nfl.csv.bz2'
    source = os.path.join(get_tests_dir(), data_file)
    df = pandas.read_csv(source)
    yield {'df': df}


def test_cleaner(data):
    from cleanset import Cleaner
    df = data['df']
    thr = 0.1
    cleaner = Cleaner(fna=thr, axis=0.5)
    assert cleaner.fit_transform(df).shape == (1868, 73)


def test_clean(data):
    from cleanset import clean
    df = data['df']
    thr = 0.1
    rows, cols = clean(df, fna=thr, axis=0.5)
    assert (len(rows), len(cols)) == (1868, 73)
