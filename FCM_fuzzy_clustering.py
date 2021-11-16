# Fuzzy C-means algorithm

import numpy as np

np.random.seed(42)


def distance_sq(x1, x2):
    dif = x1 - x2
    return np.dot(dif.T, dif)


class FCM:
    def __init__(self, K=3, max_iters=100):
        self.K = K
        self.max_iters = max_iters

        # list of sample idx for each cluster
        self.centroids = []

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        # initialize centroids
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

        # optimization
        for a in range(self.max_iters):
            # update membership matrix
            self.membership = np.array([self.membership_update_shwedow(self.centroids, sample) for sample in self.X])
            # update centroids
            centroids_old = self.centroids
            self.centroids = self.centroids_update(self.membership)
            # check if converged
            if self._is_converged(centroids_old, self.centroids):
                break
        return self.membership, self.centroids

    def centroids_update(self, membership):
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(membership.T):
            numerator = np.zeros((1, self.n_features))
            for idx, weight in enumerate(cluster):
                v = ((weight**self.n_features) * self.X[idx])
                numerator += v
            denominator = np.sum(cluster**self.n_features)
            weighted_cluster_mean = numerator/denominator
            centroids[cluster_idx] = weighted_cluster_mean
        return centroids

    def membership_update_xuanli(self, centroids, sample):
        dist_vector = []
        for centroid in centroids:
            if np.sum(sample-centroid) != 0.0:
                dist = distance_sq(sample, centroid)**(-1/(self.n_features-1))
            else:
                dist = 0.0
            dist_vector.append(dist)
        cluster_weights = (dist_vector/np.sum(dist_vector))
        return cluster_weights

    def membership_update_shwedow(self, centroids, sample):
        dist_vector = [distance_sq(sample, centroid) for centroid in centroids]
        dist_vector = [1e-9 if dist == 0.0 else dist for dist in dist_vector]
        dist_vector = np.array(dist_vector)
        cluster_weights = []
        for dist in dist_vector:
            weight = np.sum((dist/dist_vector)**(1/(self.n_features-1)))
            weight = weight**-1
            cluster_weights.append(weight)
        return cluster_weights

    def _is_converged(self, centroids_old, centroids_new):
        distances = [distance_sq(centroids_old[i], centroids_new[i]) for i in range(self.K)]
        return sum(distances) < 10e-06


def accuracy(y_true, membership):
    acc = 0.0
    for idx, memb in enumerate(membership):
        main_c = np.argmax(memb)
        main_c = main_c if main_c==0 else np.abs(main_c - 3)
        if main_c == y_true[idx]:
            acc += 1
    acc = acc/len(y_true)
    return acc


if __name__ == "__main__":
    from sklearn.datasets import make_blobs

    X, y = make_blobs(centers=3, n_samples=120, n_features=2, shuffle=True, random_state=1234)
    print(X.shape)

    clusters = len(np.unique(y))
    print(clusters)

    c = FCM(K=3, max_iters=300)
    memb_matrix, centroids = c.predict(X)

    accuracy = accuracy(y, memb_matrix)
    print(accuracy)
