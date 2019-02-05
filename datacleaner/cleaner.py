import numpy
from datacleaner.base import BaseEstimator, TransformerMixin


class Cleaner(BaseEstimator, TransformerMixin):
    """Clean data. see StandardScaler"""
    def __init__(self, condition, thr=0.1):
        self.condition = None
        self.mask_ = None
        if callable(condition):
            self.condition = condition
        elif hasattr(condition, 'ndim'):
            # check if input is a mask array
            if numpy.unique(condition) == [0, 1]:
                self.mask_ = condition
        self.thr = thr
        self.records_ = None
        self.fields_ = None

    @staticmethod
    def mask_from(condition, X):
        return numpy.array(
            [[condition(x) for x in record] for record in X])

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
        if self.mask_ is None:
            mask = self.mask_from(self.condition, X)
        self.mask_ = mask

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
                return self
            else:
                if len(records) % 1 == 0:
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
