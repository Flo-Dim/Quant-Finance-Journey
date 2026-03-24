import numpy as np


class BlackScholesModel:
    def __init__(self, S0, r, sigma, T, n_steps, n_paths):
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.T = T
        self.n_steps = n_steps
        self.n_paths = n_paths
        self.dt = T / n_steps

    def simulate_paths(self):
        # Vectorised GBM: generate all random draws at once
        Z = np.random.normal(0, 1, (self.n_paths, self.n_steps))

        increments = np.exp(
            (self.r - 0.5 * self.sigma ** 2) * self.dt
            + self.sigma * np.sqrt(self.dt) * Z
        )

        S = np.empty((self.n_paths, self.n_steps + 1))
        S[:, 0] = self.S0
        S[:, 1:] = self.S0 * np.cumprod(increments, axis=1)

        return S