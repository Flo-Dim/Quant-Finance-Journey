import numpy as np


class PhoenixAutocall:
    def __init__(
        self,
        notional,
        coupon_rate,
        coupon_barrier,
        autocall_barrier,
        protection_barrier,
        observation_indices,
        T,
        r,
        S0,
    ):
        self.notional = notional
        self.coupon_rate = coupon_rate
        self.coupon_barrier = coupon_barrier
        self.autocall_barrier = autocall_barrier
        self.protection_barrier = protection_barrier
        self.observation_indices = observation_indices
        self.T = T
        self.r = r
        self.S0 = S0

    def payoff(self, path, dt):
        pv = 0.0  # Present value accumulator

        for idx in self.observation_indices:
            S_obs = path[idx]
            t_obs = idx * dt

            # Autocall: product terminates, pay notional + coupon now 
            if S_obs >= self.autocall_barrier:
                pv += (self.notional + self.coupon_rate * self.notional) * np.exp(
                    -self.r * t_obs
                )
                return pv  # Early termination

            # Coupon: above coupon barrier but not autocalled 
            if S_obs >= self.coupon_barrier:
                # Discount each coupon back from its own observation date
                pv += self.coupon_rate * self.notional * np.exp(-self.r * t_obs)

        # Maturity: product was never autocalled 
        S_T = path[-1]

        if S_T >= self.protection_barrier:
            capital = self.notional
        else:
            # Capital-at-risk: loss proportional to underlying drop
            capital = self.notional * (S_T / self.S0)

        pv += capital * np.exp(-self.r * self.T)

        return pv