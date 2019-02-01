import numpy
from datacleaner.base import BaseEstimator, TransformerMixin


class EntrySelector:
    """Define a set of entries given some condition.

    The mask method returns a mask for undesired entries of an array.
    """
    def __init__(self, condition):
        if callable(condition):
            self.condition = condition
        else:
            if isinstance(condition, str):
                self.condition = set((condition,))
            else:
                try:
                    self.condition = set(condition)
                except TypeError:
                    self.condition = set((condition,))

    def mask_from_function(self, X):
        return numpy.array(
            [[self.condition(x) for x in record] for record in X])

    def mask_from_set(self, X):
        import functools
        return functools.reduce(
            numpy.logical_or, [X == c for c in self.condition])

    @property
    def mask(self):
        if callable(self.condition):
            return self.mask_from_condition
        else:
            return self.mask_from_set


class Cleaner(BaseEstimator, TransformerMixin, EntrySelector):
    """Clean data. see StandardScaler"""
    def __init__(self, na='NA', thr=0.1):
        super().__init__(condition=na)
        self.thr = thr
        self.records_ = None
        self.fields_ = None
        self.mask_ = None

    def fit(self, X, y=None):
        """Compute the mean and std to be used for later scaling.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        y
            Ignored
        """

        n, p = X.shape
        records = list(range(n))
        fields = list(range(p))
        mask = self.mask(X)

        n1, p1 = n, p
        ng0 = mask.sum(axis=0)  # p-dimensional (columns)
        ng1 = mask.sum(axis=1)  # n-dimensional (rows)
        while 1:

            r = numpy.argmax(ng1)  # index of most gappy row
            c = numpy.argmax(ng0)  # index of most gappy column

            nr = ng1[r]
            nc = ng0[c]

            if nr <= p1 * self.thr and nc <= n1 * self.thr:
                self.records_, self.fields_ = records, fields
                records = [1 if x in records else 0 for x in range(n)]
                fields = [1 if x in fields else 0 for x in range(p)]
                self.mask_ = numpy.outer(records, fields)
                return self
            else:
                if len(records) % 100 == 0:
                    print(len(records), n1, p1, nr/p1, nc/n1)
            if nc/n1 > nr/p1:
                # remove a column
                p1 -= 1
                fields.remove(c)
                ng0[c] = 0
                ng1 -= mask[:, c]
                mask[:, c] = False
                # ali = numpy.delete(ali, cmax, axis=1)
            else:
                n1 -= 1
                # remve a row
                records.remove(r)
                ng0 -= mask[r]
                ng1[r] = 0
                mask[r] = False
            # ali = numpy.delete(ali, rmax, axis=0)

    def transform(self, X):
        if self.records_ is not None:
            return X[self.records_][:, self.fields_]
        else:
            raise ValueError('This istance is Not fitted yet.')
