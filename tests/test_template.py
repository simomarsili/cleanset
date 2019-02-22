data_file = 'nfl.csv.bz2'


def tests_dir():
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


def test_cleanset():
    import os
    import pandas
    from cleanset import Cleaner
    source = os.path.join(tests_dir(), data_file)
    df = pandas.read_csv(source)
    thr = 0.1
    cleaner = Cleaner(f0=thr, f1=thr, axis=0.5)
    assert cleaner.fit_transform(df).shape == (1868, 73)
