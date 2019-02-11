import numpy
from cleanset.base import BaseEstimator, TransformerMixin


class Cleaner(BaseEstimator, TransformerMixin):
    """Cleaner class.

    Parameters
    ----------
    t0 : float
        Target fraction of invalid entries for rows.
    t1 : float
        Target fraction of invalid entries for columns.
    condition : callable or array
        If callable, condition(x) is True if x is an invalid value.
        If a 2D array, a boolean mask for invalid entries with shape
        [n_samples, n_features].
        Default: 'isna', detect NA values via pandas.isna() or numpy.isnan().
    axis : int or float, optional
        If axis == 0, first remove rows with too many invalid entries,
        then columns. If 0 < axis < 1, iterately remove the row/column with the
        largest fraction of invalid entries; larger values tend to remove
        columns faster than rows. If axis == 1, columns are removed first.

    """

    def __init__(self, t0, t1, *, condition='isna', axis=0.5):
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
        if t0 is None or t1 is None:
            raise ValueError('Thresholds (%r, %r) should be floats in the '
                             '0 < thr < 1 range' % (t0, t1))
        self.t0, self.t1 = t0, t1
        if 0 <= axis <= 1:
            self.axis = numpy.float(axis)
        else:
            raise ValueError('axis must be in the [0,1] range')

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

        # check axis in {0,1}
        if self.axis.is_integer():
            if self.axis == 0:
                # first remove cols
                cols = [k for k, x in enumerate(self.mask_.mean(axis=0))
                        if x <= self.t1]
                rows = [k for k, x in
                        enumerate(self.mask_[:, cols].mean(axis=1))
                        if x <= self.t0]
            elif self.axis == 1:
                # first remove rows
                rows = [k for k, x in enumerate(self.mask_.mean(axis=1))
                        if x <= self.t0]
                cols = [k for k, x in
                        enumerate(self.mask_[rows].mean(axis=0))
                        if x <= self.t1]
            self.rows_, self.cols_ = rows, cols
            return self

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

            row_fraction = (1 - self.axis) * (nr / p1)
            col_fraction = self.axis * (nc / n1)

            if nr <= p1 * self.t0:
                row_convergence = True
                row_fraction = -1
            if nc <= n1 * self.t1:
                col_convergence = True
                col_fraction = -1

            if row_convergence and col_convergence:
                self.rows_, self.cols_ = rows, cols
                return self
            if col_fraction / self.t1 > row_fraction / self.t0:
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

    def transform(self, X):
        if self.rows_ is not None:
            try:
                return X.iloc[:, self.cols_].iloc[self.rows_]
            except AttributeError:
                return X[self.rows_][:, self.cols_]
        else:
            raise ValueError('This istance is Not fitted yet.')


def clean(X, t0, t1, *, condition='isna', axis=0.5, return_clean_data=False):
    """
    Clean data from invalid entries.

    Parameters
    ----------
    X : dataframe or array-like, shape [n_samples, n_features]
        The data used to compute the valid rows and columns.
    t0 : float
        Target fraction of invalid entries for rows.
    t1 : float
        Target fraction of invalid entries for columns.
    condition : callable or array
        If callable, condition(x) is True if x is an invalid value.
        If a 2D array, a boolean mask for invalid entries with shape
        [n_samples, n_features].
        Default: 'isna', detect NA values via pandas.isna() or numpy.isnan().
    axis : int or float, optional
        If axis == 0, first remove rows with too many invalid entries,
        then columns. If 0 < axis < 1, iterately remove the row/column with the
        largest fraction of invalid entries; larger values tend to remove
        columns faster than rows. If axis == 1, columns are removed first.
    return_clean_data : bool, optional
        If True, also return filtered data.

    Returns
    -------
    (rows, columns) : tuple of lists
        Indices of rows and columns identifying a submatrix of data
        for which the fraction of invalid entries is lower than the thresholds.
        If return_clean_data is True: return (rows, columns, filtered_data)

    """
    cleaner = Cleaner(condition=condition, thr=thr, axis=axis)
    cleaner.fit(X)
    if return_clean_data:
        return cleaner.rows_, cleaner.cols_, cleaner.transform(X)
    else:
        return cleaner.rows_, cleaner.cols_