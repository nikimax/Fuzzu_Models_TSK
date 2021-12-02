import numpy as np

# data frame normalization
def normalize_df(X):
    X_norm = []
    for x in X.T:
        x = normalize_factor(x)
        X_norm.append(x)
    return np.array(X_norm).T

# variable normalization
def normalize_factor(X):
    min_idx = np.argmin(X)
    X -= X[min_idx]
    max_idx = np.argmax(X)
    X = X/X[max_idx]
    return X
