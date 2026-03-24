import numpy as np


class MonteCarloPricer:
    def __init__(self, model, product):
        self.model = model
        self.product = product

    def price(self):
        paths = self.model.simulate_paths()

        payoffs = np.array(
            [self.product.payoff(path, self.model.dt) for path in paths]
        )

        return payoffs.mean()