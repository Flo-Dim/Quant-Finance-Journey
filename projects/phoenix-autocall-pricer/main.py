from pricer.model import BlackScholesModel
from pricer.product import PhoenixAutocall
from pricer.pricer import MonteCarloPricer
from utils.market import get_market_data

import numpy as np


def main():
    # Reproducibility
    np.random.seed(42)

    # Market parameters
    S0, sigma, prices = get_market_data()
    r = 0.02
    T = 1

    print(f"Spot: {S0:.2f}")
    print(f"Volatility: {sigma:.2%}")

    # Simulation params
    n_steps = 252
    n_paths = 10_000

    # Product params
    notional = 1_000
    coupon_rate = 0.02

    coupon_barrier = 0.70 * S0
    autocall_barrier = 1.00 * S0
    protection_barrier = 0.60 * S0

    # Monthly observations (12 roughly-equal steps over 252 trading days)
    n_obs = 12
    observation_indices = np.linspace(1, n_steps, n_obs, dtype=int)

    # Build objects
    model = BlackScholesModel(S0, r, sigma, T, n_steps, n_paths)

    product = PhoenixAutocall(
        notional,
        coupon_rate,
        coupon_barrier,
        autocall_barrier,
        protection_barrier,
        observation_indices,
        T,
        r,
        S0,
    )

    pricer = MonteCarloPricer(model, product)

    # Price
    price = pricer.price()

    print(f"Phoenix Autocall Price: {price:.2f}")


if __name__ == "__main__":
    main()