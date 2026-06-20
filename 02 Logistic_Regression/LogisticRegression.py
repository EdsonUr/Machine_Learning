import numpy as np

class LogisticRegression:

    def __init__(self, lr = 0.001, n_iters = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.w = None
        self.b = None
        self.J_history = []
    
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def _compute_cost(self, X, y, m):
        z = np.dot(X, self.w) + self.b
        f_wb = self._sigmoid(z)
        f_wb = np.clip(f_wb, 1e-15, 1 - 1e-15)
        cost = (-1 / m) * np.sum(y * np.log(f_wb) + (1 - y) * np.log(1 - f_wb))
        return cost

    def _compute_gradient(self, X, y, m):
        z = np.dot(X, self.w) + self.b
        f_wb = self._sigmoid(z)

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

    def predict_probability(self, X):
        z = np.dot(X, self.w) + self.b
        return self._sigmoid(z)

    def predict(self, X):
        probabilities = self.predict_probability(X)
        return np.where(probabilities >= 0.5, 1, 0)