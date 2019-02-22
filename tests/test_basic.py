data_file = 'nfl.csv.bz2'


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


def test_cleaner():
    import os
    import pandas
    from cleanset import Cleaner
    source = os.path.join(get_tests_dir(), data_file)
    df = pandas.read_csv(source)
    thr = 0.1
    cleaner = Cleaner(f0=thr, f1=thr, axis=0.5)
    assert cleaner.fit_transform(df).shape == (1868, 73)


def test_clean():
    import os
    import pandas
    from cleanset import clean
    source = os.path.join(get_tests_dir(), data_file)
    df = pandas.read_csv(source)
    thr = 0.1
    rows, cols = clean(df, f0=thr, f1=thr, axis=0.5)
    assert (len(rows), len(cols)) == (1868, 73)
