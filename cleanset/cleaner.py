import logging
import numpy
from cleanset.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class CleansetError(Exception):
    """Base class for pcdhit exceptions."""


class InvalidEntriesDefinitionError(CleansetError):
    """Invalid definition."""


class InvalidTargetFractionError(CleansetError):
    """Invalid target fraction of invalid values."""

    def __init__(self):
        message = 'valid values are 0.0 <= target_fraction <= 1.0'
        super().__init__(message)


class AxisError(CleansetError):
    """Invalid axis value."""

    def __init__(self):
        message = 'valid values are 0 <= axis <= 1'
        super().__init__(message)


class NotFittedError(CleansetError):
    """Istance not fitted."""


class Cleaner(BaseEstimator, TransformerMixin):
    """Cleaner class.

    Parameters
    ----------
    fna : float or tuple
        Target fraction(s) of invalid entries for rows/columns.
    condition : callable or array
        If callable, condition(x) is True if x is an invalid value.
        If a 2D array, a boolean mask for invalid entries with shape
        [n_samples, n_features].
        Default: 'isna', detect NA values via pandas.isna() or numpy.isnan().
    axis : int or float, optional
        If axis == 0, first remove rows with too many invalid entries,
        then columns. If 0 < axis < 1, iterately remove the row/column with the
        largest fraction of invalid entries; values larger than 0.5 remove
        columns faster than rows. If axis == 1, columns are removed first.

    """

    def __init__(self, fna=(0.1, 0.1), *, condition='isna', axis=0.5):
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
            raise InvalidEntriesDefinitionError(condition)

        try:
            f0, f1 = fna
        except TypeError:
            f0 = f1 = fna
        if f0 is None or (not 0 < f0 < 1):
            raise InvalidTargetFractionError
        if f1 is None or (not 0 < f1 < 1):
            raise InvalidTargetFractionError
        self.fna = (f0, f1)
        if 0 <= axis <= 1:
            self.axis = numpy.float(axis)
        else:
            raise AxisError

    @staticmethod
    def _get_mask(X, condition):
        try:
            # check if dataframe
            # TODO: applymap
            return X.apply(condition, result_type='broadcast').values
        except AttributeError:
            return numpy.vectorize(condition)(X)

    def _fit_remove_cols_first(self):
        # first remove cols
        self.cols_ = [
            k for k, x in enumerate(self.mask_.mean(axis=0))
            if x <= self.fna[1]
        ]
        if not self.cols_:
            self.rows_ = []
            return self
        self.rows_ = [
            k for k, x in enumerate(self.mask_[:, self.cols_].mean(axis=1))
            if x <= self.fna[0]
        ]
        return self

    def _fit_remove_rows_first(self):
        # first remove rows
        self.rows_ = [
            k for k, x in enumerate(self.mask_.mean(axis=1))
            if x <= self.fna[0]
        ]
        if not self.rows_:
            self.cols_ = []
            return self
        self.cols_ = [
            k for k, x in enumerate(self.mask_[self.rows_].mean(axis=0))
            if x <= self.fna[1]
        ]
        return self

    def _remove_column(self, c):
        # remove a column
        self.cols_.remove(c)
        self.col_ninvalid[c] = 0
        self.row_ninvalid -= self.mask_[:, c]

    def _remove_rows(self, r):
        # remove all rows with the same number of invalid entries of row r
        nr = self.row_ninvalid[r]
        rset = [x for x in self.rows_ if self.row_ninvalid[x] == nr]
        self.rows_ = [x for x in self.rows_ if self.row_ninvalid[x] < nr]
        self.row_ninvalid[rset] = 0
        self.col_ninvalid -= self.mask_[rset].sum(axis=0)

    def fit(self, X, y=None):
        """Compute a subset of rows and columns.

        Parameters
        ----------
        X : dataframe or array-like, shape [n_samples, n_features]
            The data used to compute the valid rows and columns.
        y
            Ignored
        """

        f0, f1 = self.fna
        n, p = X.shape
        self.rows_ = list(range(n))
        self.cols_ = list(range(p))

        # build the mask
        if self.mask_ is None:
            self.mask_ = self._get_mask(X, self.condition)

        # check axis in {0,1}
        if self.axis == 1:
            return self._fit_remove_cols_first()
        elif self.axis == 0:
            return self._fit_remove_rows_first()

        self.col_ninvalid = self.mask_.sum(axis=0)  # p-dimensional (columns)
        self.row_ninvalid = self.mask_.sum(axis=1)  # n-dimensional (rows)

        while 1:
            n1 = len(self.rows_)
            p1 = len(self.cols_)
            if not p1 or not n1:
                self.rows_ = []
                self.cols_ = []
                return self

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

            if nr <= p1 * f0:
                row_fraction = -1
            if nc <= n1 * f1:
                col_fraction = -1

            if row_fraction == -1 and col_fraction == -1:
                return self

            if col_fraction / f1 > row_fraction / f0:
                self._remove_column(c)
            else:
                self._remove_rows(r)

    def transform(self, X):
        if self.rows_ is not None:
            try:
                return X.iloc[:, self.cols_].iloc[self.rows_]
            except AttributeError:
                return X[self.rows_][:, self.cols_]
        else:
            raise NotFittedError


def clean(X, fna=(0.1, 0.1), *, condition='isna', axis=0.5,
          return_clean_data=False):
    """
    Clean data from invalid entries.

    Parameters
    ----------
    X : dataframe or array-like, shape [n_samples, n_features]
        The data used to compute the valid rows and columns.
    fna : tuple
        Target fractions of invalid entries for rows/columns.
    condition : callable or array
        If callable, condition(x) is True if x is an invalid value.
        If a 2D array, a boolean mask for invalid entries with shape
        [n_samples, n_features].
        Default: 'isna', detect NA values via pandas.isna() or numpy.isnan().
    axis : int or float, optional
        If axis == 0, first remove rows with too many invalid entries,
        then columns. If 0 < axis < 1, iterately remove the row/column with the
        largest fraction of invalid entries; values larger than 0.5 remove
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
    cleaner = Cleaner(fna=fna, condition=condition, axis=axis)
    cleaner.fit(X)
    if return_clean_data:
        return cleaner.rows_, cleaner.cols_, cleaner.transform(X)
    else:
        return cleaner.rows_, cleaner.cols_
