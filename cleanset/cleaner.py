import numpy
from cleanset.base import BaseEstimator, TransformerMixin


class Cleaner(BaseEstimator, TransformerMixin):
    """Cleaner class.

    Parameters
    ----------
    condition : callable or array
        If callable, condition(x) is True if x is an invalid value.
        If a 2D boolean array, a mask for invalid entries
        (with shape identical to data).
    thr : tuple or int, optional
        The desired ratio of invalid entries.
        If a single integer, use the same value both for rows and columns.
    alpha : float, optional
        For 0.5 < alpha < 1, remove rows more easily than columns.
        0 < alpha < 1.

    """

    def __init__(self, condition, thr=0.1, alpha=0.5):
        self.condition = None
        self.mask_ = None
        self.rows_ = None
        self.cols_ = None
        self.col_invalid = None
        self.row_invalid = None
        if callable(condition):
            self.condition = condition
        elif hasattr(condition, 'ndim'):
            if numpy.unique(condition) == [0, 1]:
                self.mask_ = condition
        try:
            self.row_thr, self.col_thr = thr
        except TypeError:
            self.row_thr, self.col_thr = (thr, thr)
        if 0 < alpha < 1:
            self.alpha = alpha
        else:
            raise ValueError('alpha must be in the [0,1] range')

    @staticmethod
    def mask_from(condition, X):
        return numpy.array(
            [[condition(x) for x in row] for row in X])

    def fit(self, X, y=None):
        """Compute the subset of valid rows and columns.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to compute the valid rows and columns.
        y
            Ignored
        """

        n, p = X.shape
        rows = list(range(n))
        cols = list(range(p))
        if self.mask_ is None:
            mask = self.mask_from(self.condition, X)
        self.mask_ = mask

        n1, p1 = n, p  # # of filtered rows and columns
        self.col_invalid = mask.sum(axis=0)  # p-dimensional (columns)
        self.row_invalid = mask.sum(axis=1)  # n-dimensional (rows)
        while 1:
            # index of the row with the largest number of invalid entries
            r = numpy.argmax(self.row_invalid)
            # index of the column with the largest number of invalid entries
            c = numpy.argmax(self.col_invalid)

            # fraction of invalid entries in row r
            nr = self.row_invalid[r] / p1
            # fraction of invalid entries in column c
            nc = self.col_invalid[c] / n1

            if nr <= self.row_thr and nc <= self.col_thr:
                self.rows_, self.cols_ = rows, cols
                print('final: ', len(rows), n1, p1, nr, nc)
                return self
            else:
                if len(rows) % 1 == 0:
                    print(len(rows), n1, p1, nr, nc)
            if (1 - self.alpha) * nc / self.col_thr > (
                    self.alpha * nr / self.row_thr):
                # remove a column
                p1 -= 1
                cols.remove(c)
                self.col_invalid[c] = 0
                self.row_invalid -= mask[:, c]
                mask[:, c] = False
            else:
                n1 -= 1
                # remve a row
                rows.remove(r)
                self.col_invalid -= mask[r]
                self.row_invalid[r] = 0
                mask[r] = False

    def transform(self, X):
        if self.rows_ is not None:
            return X[self.rows_][:, self.cols_]
        else:
            raise ValueError('This istance is Not fitted yet.')


def clean(X, condition, *, thr=0.1, alpha=0.5):
    """
    Clean data from invalid entries.

    Parameters
    ----------
    X : array-like, shape [n_samples, n_features]
        The data used to compute the valid rows and columns.
    condition : callable or array, optional
        If callable, condition(x) is True if x is an invalid value.
        If a 2D boolean array, a mask for invalid entries
        (with shape identical to data).
    thr : tuple or int, optional
        The desired ratio of invalid entries.
        If a single integer, use the same value both for rows and columns.
    alpha : float, optional
        For 0.5 < alpha < 1, remove rows more easily than columns.
        0 < alpha < 1.

    Returns
    -------
    (rows, columns) : tuple of lists
        Indices of rows and columns identifying a submatrix of data
        for which the fraction of invalid entries is lower than the thresholds.

    """
    cleaner = Cleaner(condition, thr=thr, alpha=alpha)
    cleaner.fit(X)
    return cleaner.rows_, cleaner.cols_
