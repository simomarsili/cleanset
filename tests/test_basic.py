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


prms = [
    (0., 0.2, (438, 83)),
    (0.2, 0.1, (820, 82)),
    (0.5, 0.1, (1868, 73)),
    (1., 0.1, (1869, 72))]


@pytest.mark.parametrize('axis, thr, expected', prms)
def test_cleaner(data, axis, thr, expected):
    from cleanset import Cleaner
    df = data['df']
    cleaner = Cleaner(fna=thr, axis=axis)
    assert cleaner.fit_transform(df).shape == expected


@pytest.mark.parametrize('axis, thr, expected', prms)
def test_clean(data, axis, thr, expected):
    from cleanset import clean
    df = data['df']
    rows, cols = clean(df, fna=thr, axis=axis)
    assert (len(rows), len(cols)) == expected
