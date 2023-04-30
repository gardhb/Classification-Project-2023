import numpy as np

class GaussianMixtureModel:
    def __init__(self, n_components=2, max_iter=100, tol=1e-3):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol

    def _initialize(self, X):
        n_samples, n_features = X.shape
        self.weights_ = np.ones(self.n_components) / self.n_components
        self.means_ = X[np.random.choice(n_samples, self.n_components, replace=False)]
        self.covariances_ = np.array([np.eye(n_features) for _ in range(self.n_components)])

    def _e_step(self, X):
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            diff = X - self.means_[k]
            mult = np.dot(diff, np.linalg.inv(self.covariances_[k]))
            exp = np.exp(-0.5 * np.sum(mult * diff, axis=1))
            responsibilities[:, k] = self.weights_[k] * np.sqrt(np.linalg.det(self.covariances_[k])) * exp

        responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)
        return responsibilities

    def _m_step(self, X, responsibilities):
        n_samples = X.shape[0]
        weights_sum = np.sum(responsibilities, axis=0)

        self.weights_ = weights_sum / n_samples
        self.means_ = np.dot(responsibilities.T, X) / weights_sum[:, np.newaxis]

        for k in range(self.n_components):
            diff = X - self.means_[k]
            self.covariances_[k] = np.dot(responsibilities[:, k] * diff.T, diff) / weights_sum[k]

    def _compute_log_likelihood(self, X):
        log_likelihood = 0

        for k in range(self.n_components):
            diff = X - self.means_[k]
            mult = np.dot(diff, np.linalg.inv(self.covariances_[k]))
            exp = np.exp(-0.5 * np.sum(mult * diff, axis=1))
            log_likelihood += self.weights_[k] * np.sqrt(np.linalg.det(self.covariances_[k])) * exp

        return np.sum(np.log(log_likelihood))

    def fit(self, X):
        self._initialize(X)
        log_likelihood = -np.inf

        for _ in range(self.max_iter):
            responsibilities = self._e_step(X)
            self._m_step(X, responsibilities)

            new_log_likelihood = self._compute_log_likelihood(X)
            if np.abs(new_log_likelihood - log_likelihood) < self.tol:
                break

            log_likelihood = new_log_likelihood

    def predict(self, X):
        return np.argmax(self._e_step(X), axis=1)