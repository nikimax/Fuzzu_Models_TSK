import numpy as np
import copy
from itertools import *


def MSE(y_true, y_pred):
    return np.mean((y_true - y_pred)**2, dtype=np.float64)


def weight_mult(c, weights):
    r = copy.deepcopy(c)
    for idx, num in enumerate(weights):
        r[idx] = r[idx] * num
    return r


def power_factors(X, deg):
    X_power = []
    for x in X:
        x_power = x**deg
        X_power.append(x_power)
    return np.array(X_power)


def normalise_x(X):
    X_norm = []
    for x in X.T:
        min_idx = np.argmin(x)
        x -= x[min_idx]
        max_idx = np.argmax(x)
        x = x/x[max_idx]
        X_norm.append(x)
    return np.array(X_norm).T


def normalise_y(X):
    min_idx = np.argmin(X)
    X -= X[min_idx]
    max_idx = np.argmax(X)
    X = X/X[max_idx]
    return X


class TakagiSugeno:
    def __init__(self, cluster_n=2, lr=0.01, n_iters=1500):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.weights_best = None
        self.bias_best = None
        self.combination_best = None
        self.cluster_n = cluster_n

    def fit(self, X, y, cluster_w):
        power_degree = np.arange(self.cluster_n)
        power_degree += 1
        models_list = [[], [], [], []]
        for combination in permutations(power_degree):
            X_polynom = []
            for c in combination:
                X_power = power_factors(X, c)
                X_polynom.append(X_power)
            self.model_estimation(X_polynom, y, cluster_w)
            y_pred = self.y_estimation(X_polynom, cluster_w, self.weights, self.bias)
            mse = MSE(y, y_pred)
            models_list[0].append(copy.deepcopy(self.weights))
            models_list[1].append(copy.deepcopy(self.bias))
            models_list[2].append(mse)
            models_list[3].append(combination)

        best_model = np.argmin(models_list[2])
        self.weights_best = models_list[0][best_model]
        self.bias_best = models_list[1][best_model]
        self.combination_best = models_list[3][best_model]
        #return models_list

    def model_estimation(self, X_polynom, y, cluster_w):
        n_samples, n_features = X_polynom[0].shape
        self.weights = np.zeros((self.cluster_n, n_features))
        self.bias = np.zeros(self.cluster_n)

        for _ in range(self.n_iters):
            y_predicted = self.y_estimation(X_polynom, cluster_w, self.weights, self.bias)

            for c in range(self.cluster_n):
                # multiple grad count
                dw = (2 / n_samples) * np.dot(weight_mult(X_polynom[c], cluster_w[c]).T, (y_predicted - y))
                db = (2 / n_samples) * np.sum(weight_mult((y_predicted - y), cluster_w[c]))
                # weights update
                self.weights[c] -= self.lr * dw
                self.bias[c] -= self.lr * db

    def y_estimation(self, X_polynom, cluster_w, weights, bias):
        y_predicted = np.zeros(len(X_polynom[0]))
        for c in range(self.cluster_n):
            # evaluate y
            y_pred_cluster = np.dot(X_polynom[c], weights[c]) + bias[c]
            weighted_y_pred = weight_mult(y_pred_cluster, cluster_w[c])
            y_predicted += weighted_y_pred
        return y_predicted

    def predict(self, X, cluster_w):
        X_polynom = []
        for c in self.combination_best:
            X_power = power_factors(X, c)
            X_polynom.append(X_power)
        y_predicted = self.y_estimation(X_polynom, cluster_w, self.weights_best, self.bias_best)
        return y_predicted


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from sklearn import datasets

    # Prepare data
    X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=11, noise=20, random_state=1)
    # Normalisation
    X_norm = np.array(normalise_x(X_numpy), dtype=np.float64)
    y_norm = np.array(normalise_y(y_numpy), dtype=np.float64)
    # Create y
    y_sq = power_factors(y_norm, 2)
    y = y_norm * 0.6 + y_sq * 0.4
    # Create membership matrix
    membership = np.zeros((len(X_norm), 2))
    membership[:, 0] = 0.6
    membership[:, 1] = 0.4
    membership = np.array(membership, dtype=np.float64)
    membership = membership.T

    # training loop
    model = TakagiSugeno(lr=0.1, n_iters=1200)
    model.fit(X_norm, y, membership)
    y_pred = model.predict(X_norm, membership)

    plt.plot(y, color='r')
    plt.plot(y_pred, color='blue')
    plt.show()

    print("MSE", MSE(y, y_pred))
