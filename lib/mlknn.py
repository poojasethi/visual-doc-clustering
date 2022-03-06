import scipy.sparse as sparse
from sklearn.neighbors import NearestNeighbors
from skmultilearn.adapt.mlknn import MLkNN as BackwardsCompatibleMLkNN
from skmultilearn.utils import get_matrix_in_format


class MLkNN(BackwardsCompatibleMLkNN):
    def _compute_cond(self, X, y):
        """
        NOTE(pooja): This function monkey patches _compute_cond. It
        is nearly identical to that of _compute_cond in the original
        MLkNN implementation. The *only* change I have made is the
        initialization of self.knn_.

        Now:
        self.knn_ = NearestNeighbors(n_neighbors=self.k).fit(X)

        Before:
        self.knn_ = NearestNeighbors(self.k).fit(X)

        This change is necessary do to a change in the sklearn's
        NearestNeighbors class, i.e., it now expects n_neighbors
        to be passed in as a keyword arg.

        Also see the open issue here:
        #231: https://github.com/scikit-multilearn/scikit-multilearn/pull/231
        #224: https://github.com/scikit-multilearn/scikit-multilearn/issues/224
        ---

        Helper function to compute for the posterior probabilities

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse
            input features, can be a dense or sparse matrix of size
            :code:`(n_samples, n_features)`
        y : numpy.ndaarray or scipy.sparse {0,1}
            binary indicator matrix with label assignments.

        Returns
        -------
        numpy.ndarray
            the posterior probability given true
        numpy.ndarray
            the posterior probability given false
        """
        self.knn_ = NearestNeighbors(n_neighbors=self.k).fit(X)
        c = sparse.lil_matrix((self._num_labels, self.k + 1), dtype="i8")
        cn = sparse.lil_matrix((self._num_labels, self.k + 1), dtype="i8")

        label_info = get_matrix_in_format(y, "dok")

        neighbors = [
            a[self.ignore_first_neighbours :]
            for a in self.knn_.kneighbors(X, self.k + self.ignore_first_neighbours, return_distance=False)
        ]

        for instance in range(self._num_instances):
            deltas = label_info[neighbors[instance], :].sum(axis=0)
            for label in range(self._num_labels):
                if label_info[instance, label] == 1:
                    c[label, deltas[0, label]] += 1
                else:
                    cn[label, deltas[0, label]] += 1

        c_sum = c.sum(axis=1)
        cn_sum = cn.sum(axis=1)

        cond_prob_true = sparse.lil_matrix((self._num_labels, self.k + 1), dtype="float")
        cond_prob_false = sparse.lil_matrix((self._num_labels, self.k + 1), dtype="float")
        for label in range(self._num_labels):
            for neighbor in range(self.k + 1):
                cond_prob_true[label, neighbor] = (self.s + c[label, neighbor]) / (
                    self.s * (self.k + 1) + c_sum[label, 0]
                )
                cond_prob_false[label, neighbor] = (self.s + cn[label, neighbor]) / (
                    self.s * (self.k + 1) + cn_sum[label, 0]
                )
        return cond_prob_true, cond_prob_false
