import numpy
from cleanset.base import BaseEstimator, TransformerMixin


class Cleaner(BaseEstimator, TransformerMixin):
    """Cleaner class.

    Parameters
    ----------
    condition : callable or array
        If callable, condition(x) is True if x is an invalid value.
        If a 2D array, a boolean mask for invalid entries with shape
        [n_samples, n_features].
        Default: 'isna', detect NA values via pandas.isna() or numpy.isnan().
    thr : tuple or int, optional
        Target fraction of invalid entries for rows and columns.
        If a single integer, use the same value.
    alpha : float, optional
        Larger values bias the filtering process toward row and against column
        removal. 0 < alpha < 1.

    """

    def __init__(self, condition='isna', thr=0.1, alpha=0.5):
        self.mask_ = None
        self.rows_ = None
        self.cols_ = None
        self.col_ninvalid = None
        self.row_ninvalid = None
        if condition == 'isna':
            try:
                import pandas
            except ImportError:
                # use numpy
                def condition(x):
                    try:
                        return numpy.isnan(x)
                    except TypeError:
                        return False
                self.condition = condition
            else:
                self.condition = pandas.isna
        elif callable(condition):
            self.condition = condition
        elif hasattr(condition, 'ndim'):
            if set(numpy.unique(condition)) == set([0, 1]):
                self.mask_ = condition
        else:
            raise ValueError('Invalid condition: %r' % condition)
        try:
            self.row_thr, self.col_thr = thr
        except TypeError:
            self.row_thr, self.col_thr = (thr, thr)
        if 0 < alpha < 1:
            self.alpha = alpha
        else:
            raise ValueError('alpha must be in the [0,1] range')

    def fit(self, X, y=None):
        """Compute the subset of valid rows and columns.

        Parameters
        ----------
        X : dataframe or array-like, shape [n_samples, n_features]
            The data used to compute the valid rows and columns.
        y
            Ignored
        """

        n, p = X.shape
        rows = list(range(n))
        cols = list(range(p))

        # build the mask
        if self.mask_ is None:
            try:
                # check if dataframe
                self.mask_ = X.apply(
                    self.condition, result_type='broadcast').values
            except AttributeError:
                self.mask_ = numpy.vectorize(self.condition)(X)

        n1, p1 = n, p  # # of filtered rows and columns
        self.col_ninvalid = self.mask_.sum(axis=0)  # p-dimensional (columns)
        self.row_ninvalid = self.mask_.sum(axis=1)  # n-dimensional (rows)
        row_convergence = False
        col_convergence = False
        while 1:
            # index of the row with the largest number of invalid entries
            r = numpy.argmax(self.row_ninvalid)
            # index of the column with the largest number of invalid entries
            c = numpy.argmax(self.col_ninvalid)

            # n. of invalid entries in row r
            nr = self.row_ninvalid[r]
            # n. of invalid entries in column c
            nc = self.col_ninvalid[c]

            col_fraction = (1 - self.alpha) * (nc / n1)
            row_fraction = self.alpha * (nr / p1)

            if nr <= p1 * self.row_thr:
                row_convergence = True
                row_fraction = -1
            if nc <= n1 * self.col_thr:
                col_convergence = True
                col_fraction = -1

            # print(n1, p1, row_fraction, col_fraction)
            if row_convergence and col_convergence:
                self.rows_, self.cols_ = rows, cols
                return self
            if col_fraction / self.col_thr > row_fraction / self.row_thr:
                # remove a column
                p1 -= 1
                cols.remove(c)
                self.col_ninvalid[c] = 0
                self.row_ninvalid -= self.mask_[:, c]
            else:
                rset = [x for x in rows if self.row_ninvalid[x] == nr]
                nrset = len(rset)
                # remove all rows with the same number of invalid entries
                n1 -= nrset
                rows = [x for x in rows if self.row_ninvalid[x] < nr]
                self.row_ninvalid[rset] = 0
                self.col_ninvalid -= self.mask_[rset].sum(axis=0)
            assert n1 > 0 and p1 > 0, 'This point should not be reached'

    def transform(self, X):
        if self.rows_ is not None:
            try:
                return X.iloc[:, self.cols_].iloc[self.rows_]
            except AttributeError:
                return X[self.rows_][:, self.cols_]
        else:
            raise ValueError('This istance is Not fitted yet.')


def clean(X, *, condition='isna', thr=0.1, alpha=0.5, return_clean_data=False):
    """
    Clean data from invalid entries.

    Parameters
    ----------
    X : dataframe or array-like, shape [n_samples, n_features]
        The data used to compute the valid rows and columns.
    condition : callable or array
        If callable, condition(x) is True if x is an invalid value.
        If a 2D array, a boolean mask for invalid entries with shape
        [n_samples, n_features].
        Default: 'isna', detect NA values via pandas.isna() or numpy.isnan().
    thr : tuple or int, optional
        Target fraction of invalid entries for rows and columns.
        If a single integer, use the same value.
    alpha : float, optional
        Larger values bias the filtering process toward row and against column
        removal. 0 < alpha < 1.
    return_clean_data : bool, optional
        If True, also return filtered data.

    Returns
    -------
    (rows, columns) : tuple of lists
        Indices of rows and columns identifying a submatrix of data
        for which the fraction of invalid entries is lower than the thresholds.
        If return_clean_data is True: return (rows, columns, filtered_data)

    """
    cleaner = Cleaner(condition=condition, thr=thr, alpha=alpha)
    cleaner.fit(X)
    if return_clean_data:
        return cleaner.rows_, cleaner.cols_, cleaner.transform(X)
    else:
        return cleaner.rows_, cleaner.cols_
