import numpy as np

class LinearRegression:

    def __init__(self, lr = 0.001, n_iters = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.w = None
        self.b = None
        self.J_history = []

    def _compute_cost(self, X, y, m):
        f_wb = np.dot(X, self.w) + self.b
        errors = f_wb - y
        cost = (1 / (2 * m)) * np.sum(errors ** 2)
        return cost
    
    def _compute_gradient(self, X, y, m):
        f_wb = np.dot(X, self.w) + self.b
        errors = f_wb - y

        dj_dw = (1 / m) * np.dot(X.T, errors)
        dj_db = (1 / m) * np.sum(errors)

        return dj_dw, dj_db

    def fit(self, X, y):
        m, n = X.shape

        self.w = np.zeros(n)
        self.b = 0
        self.J_history = []

        for _ in range(self.n_iters):
            dj_dw, dj_db = self._compute_gradient(X, y, m)

            self.w = self.w - self.lr * dj_dw
            self.b = self.b - self.lr * dj_db

            cost = self._compute_cost(X, y, m)
            self.J_history.append(cost)

        return self

    def predict(self, X):
        return np.dot(X, self.w) + self.b