"""
playground.py — Phoenix Autocall interactive pricer
Run with:  python playground.py
Then open the local URL (or the public share link printed in the terminal).

This file is self-contained
"""

import numpy as np
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.patches import FancyBboxPatch
from pricer.model import BlackScholesModel
from pricer.product import PhoenixAutocall
from pricer.pricer import MonteCarloPricer

# CONSTANTS  (v1 — no dividends, flat vol, constant rates)
# These are shown verbatim in the UI so users know exactly what is fixed.
CONSTANTS = {
    "Underlying":        "EURO STOXX 50 (^STOXX50E)",
    "Pricing model":     "Black-Scholes GBM",
    "Dividends":         "Not modelled (v1)",
    "Vol surface":       "Flat (single σ)",
    "Rate curve":        "Flat risk-free rate r",
    "Random seed":       "42  (reproducible)",
    "Trading days / yr": "252",
    "Observation dates": "12 monthly (equally spaced)",
    "Memory feature":    "No  (coupons lost below barrier)",
    "Fair coupon (σ=18%)": "≈ 0.14%/obs  (≈ 1.6% ann.) at autocall=100%",
}

PALETTE = {
    "bg":       "#0f1117",
    "card":     "#181c27",
    "border":   "#2a2f3e",
    "accent":   "#4f8ef7",
    "green":    "#34d399",
    "amber":    "#fbbf24",
    "red":      "#f87171",
    "text":     "#e2e8f0",
    "muted":    "#64748b",
}


# PRICING FUNCTION

def run_pricer(
    S0, sigma_pct, r_pct, T,
    notional, coupon_pct,
    autocall_pct, coupon_barrier_pct, protection_pct,
    n_paths, n_steps,
):
    # barrier sanity checks 
    errors = []
    if autocall_pct < 100:
        errors.append(
            f"\u274c  Autocall barrier ({autocall_pct}%) must be \u2265 100% of S\u2080. "
            "Below spot it fires on every path at M1 and collapses to a trivial 1-month bond."
        )
    if coupon_barrier_pct >= autocall_pct:
        errors.append(
            f"\u274c  Coupon barrier ({coupon_barrier_pct}%) must be strictly below "
            f"autocall barrier ({autocall_pct}%). They cannot fire at the same level."
        )
    if protection_pct >= coupon_barrier_pct:
        errors.append(
            f"\u274c  Protection barrier ({protection_pct}%) must be below "
            f"coupon barrier ({coupon_barrier_pct}%)."
        )
    if errors:
        empty = plt.figure()
        plt.close("all")
        return empty, empty, empty, "\n\n".join(errors)

    np.random.seed(42)

    sigma   = sigma_pct / 100
    r       = r_pct / 100
    coupon  = coupon_pct / 100

    coupon_barrier     = coupon_barrier_pct / 100 * S0
    autocall_barrier   = autocall_pct / 100 * S0
    protection_barrier = protection_pct / 100 * S0

    n_obs = 12
    observation_indices = np.linspace(1, n_steps, n_obs, dtype=int)

    model = BlackScholesModel(S0, r, sigma, T, n_steps, n_paths)
    paths = model.simulate_paths()

    product = PhoenixAutocall(
        notional, coupon, coupon_barrier, autocall_barrier,
        protection_barrier, observation_indices, T, r, S0,
    )
    pricer = MonteCarloPricer(model, product)

    # price 
    payoffs = np.array([product.payoff(p, model.dt) for p in paths])
    price   = payoffs.mean()

    # barrier levels for plots 
    obs_idx = observation_indices
    obs_t   = obs_idx * model.dt

    # autocall / coupon statistics 
    autocall_count = 0
    coupon_counts  = np.zeros(n_obs)
    autocall_at    = np.zeros(n_obs)

    for path in paths:
        called = False
        for k, idx in enumerate(obs_idx):
            S_obs = path[idx]
            if S_obs >= autocall_barrier:
                autocall_at[k] += 1
                autocall_count += 1
                called = True
                break
            elif S_obs >= coupon_barrier:
                coupon_counts[k] += 1
        _ = called  # noqa

    autocall_prob = autocall_count / n_paths * 100
    capital_loss_prob = np.mean(paths[:, -1] < protection_barrier) * 100

    # FIGURE 1 — sample paths + barrier overlay
    fig1, ax1 = plt.subplots(figsize=(9, 4.2))
    fig1.patch.set_facecolor(PALETTE["bg"])
    ax1.set_facecolor(PALETTE["card"])

    t_axis = np.linspace(0, T, n_steps + 1)
    sample = paths[:min(120, n_paths)]
    for path in sample:
        ax1.plot(t_axis, path, color=PALETTE["accent"], alpha=0.08, linewidth=0.6)

    # Median path
    ax1.plot(t_axis, np.median(paths, axis=0),
             color=PALETTE["green"], linewidth=1.8, label="Median path", zorder=5)

    # Barriers
    ax1.axhline(autocall_barrier,   color=PALETTE["amber"],  linewidth=1.2,
                linestyle="--", label=f"Autocall {autocall_pct:.0f}%")
    ax1.axhline(coupon_barrier,     color=PALETTE["accent"],  linewidth=1.2,
                linestyle=":",  label=f"Coupon {coupon_barrier_pct:.0f}%")
    ax1.axhline(protection_barrier, color=PALETTE["red"],    linewidth=1.2,
                linestyle="-.", label=f"Protection {protection_pct:.0f}%")

    # Observation ticks
    for t in obs_t:
        ax1.axvline(t, color=PALETTE["muted"], alpha=0.25, linewidth=0.6)

    ax1.set_xlabel("Time (years)", color=PALETTE["muted"], fontsize=9)
    ax1.set_ylabel("Spot level", color=PALETTE["muted"], fontsize=9)
    ax1.set_title("Monte Carlo paths  ·  barrier levels", color=PALETTE["text"],
                  fontsize=11, pad=10)
    ax1.tick_params(colors=PALETTE["muted"], labelsize=8)
    for spine in ax1.spines.values():
        spine.set_color(PALETTE["border"])
    leg = ax1.legend(fontsize=8, framealpha=0.3, labelcolor=PALETTE["text"])
    leg.get_frame().set_facecolor(PALETTE["card"])
    fig1.tight_layout(pad=1.4)

    # FIGURE 2 — payoff distribution
    fig2, ax2 = plt.subplots(figsize=(9, 3.8))
    fig2.patch.set_facecolor(PALETTE["bg"])
    ax2.set_facecolor(PALETTE["card"])

    ax2.hist(payoffs, bins=60, color=PALETTE["accent"], alpha=0.7, edgecolor="none")
    ax2.axvline(price, color=PALETTE["green"], linewidth=2,
                label=f"Mean PV = {price:.2f}")
    ax2.axvline(notional, color=PALETTE["amber"], linewidth=1.4,
                linestyle="--", label=f"Notional = {notional:.0f}")

    ax2.set_xlabel("Discounted payoff", color=PALETTE["muted"], fontsize=9)
    ax2.set_ylabel("Frequency", color=PALETTE["muted"], fontsize=9)
    ax2.set_title("Payoff distribution (all paths)", color=PALETTE["text"],
                  fontsize=11, pad=10)
    ax2.tick_params(colors=PALETTE["muted"], labelsize=8)
    for spine in ax2.spines.values():
        spine.set_color(PALETTE["border"])
    leg2 = ax2.legend(fontsize=8, framealpha=0.3, labelcolor=PALETTE["text"])
    leg2.get_frame().set_facecolor(PALETTE["card"])
    fig2.tight_layout(pad=1.4)

    # FIGURE 3 — autocall / coupon hit rates by observation date
    fig3, ax3 = plt.subplots(figsize=(9, 3.6))
    fig3.patch.set_facecolor(PALETTE["bg"])
    ax3.set_facecolor(PALETTE["card"])

    x = np.arange(n_obs)
    w = 0.38
    ax3.bar(x - w/2, autocall_at / n_paths * 100, width=w,
            color=PALETTE["amber"], alpha=0.85, label="Autocall triggered")
    ax3.bar(x + w/2, coupon_counts / n_paths * 100, width=w,
            color=PALETTE["green"], alpha=0.85, label="Coupon paid (no autocall)")

    ax3.set_xticks(x)
    ax3.set_xticklabels([f"M{i+1}" for i in range(n_obs)],
                        fontsize=7.5, color=PALETTE["muted"])
    ax3.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax3.set_ylabel("% of paths", color=PALETTE["muted"], fontsize=9)
    ax3.set_title("Autocall & coupon hit rates per observation date",
                  color=PALETTE["text"], fontsize=11, pad=10)
    ax3.tick_params(colors=PALETTE["muted"], labelsize=8)
    for spine in ax3.spines.values():
        spine.set_color(PALETTE["border"])
    leg3 = ax3.legend(fontsize=8, framealpha=0.3, labelcolor=PALETTE["text"])
    leg3.get_frame().set_facecolor(PALETTE["card"])
    fig3.tight_layout(pad=1.4)

    # ── summary markdown ───────────────────────────────────────────────────
    summary = f"""
### Results

| Metric | Value |
|---|---|
| **Phoenix Autocall Price** | **{price:,.2f}** |
| Notional | {notional:,.0f} |
| Price / Notional | {price/notional:.2%} |
| Autocall probability | {autocall_prob:.1f}% |
| Capital-loss probability | {capital_loss_prob:.1f}% |
| Paths simulated | {n_paths:,} |

> Coupons are individually discounted from each observation date.  
> Capital loss kicks in only if spot at maturity is below the protection barrier.
"""

    return fig1, fig2, fig3, summary

# UI HELPERS

def constants_markdown():
    rows = "\n".join(
        f"| {k} | `{v}` |" for k, v in CONSTANTS.items()
    )
    return f"""### Model assumptions (v1 — fixed)\n| Parameter | Value |\n|---|---|\n{rows}"""



# GRADIO APP

THEME = gr.themes.Base(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("IBM Plex Mono"), "monospace"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "monospace"],
).set(
    body_background_fill="#0f1117",
    body_text_color="#e2e8f0",
    block_background_fill="#181c27",
    block_border_color="#2a2f3e",
    block_label_text_color="#94a3b8",
    input_background_fill="#0f1117",
    input_border_color="#2a2f3e",
    slider_color="#4f8ef7",
    button_primary_background_fill="#4f8ef7",
    button_primary_background_fill_hover="#6ba3f9",
    button_primary_text_color="#ffffff",
)

with gr.Blocks(theme=THEME, title="Phoenix Autocall Pricer") as demo:

    gr.Markdown("""
# Phoenix Autocall — Interactive Pricer
**Monte Carlo · Black-Scholes · v1**  
Adjust the sliders, hit **Price**, and explore how each parameter moves the fair value.
""")

    # fixed assumptions panel 
    with gr.Accordion("📋  Model assumptions & fixed constants (v1)", open=False):
        gr.Markdown(constants_markdown())

    gr.Markdown("---")

    # inputs 
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("#### Market")
            S0    = gr.Slider(2000, 6000,  value=4500, step=50,   label="Spot S₀")
            sigma = gr.Slider(5,    60,    value=18,   step=0.5,  label="Volatility σ  (%)")
            r     = gr.Slider(0,    8,     value=2,    step=0.1,  label="Risk-free rate r  (%)")
            T     = gr.Slider(0.5,  3.0,   value=1.0,  step=0.25, label="Maturity T  (years)")

        with gr.Column(scale=1):
            gr.Markdown("#### Product")
            notional = gr.Slider(100,  10000, value=1000, step=100,  label="Notional")
            coupon   = gr.Slider(0.05, 1.0,   value=0.14, step=0.01, label="Coupon rate  (% per obs.)  — fair ≈ 0.14%")
            autocall = gr.Slider(100,  130,   value=100,  step=1,    label="Autocall barrier  (% of S₀)  [≥ 100% enforced]")
            coupon_b = gr.Slider(50,   100,   value=70,   step=1,    label="Coupon barrier  (% of S₀)")
            protect  = gr.Slider(40,   90,    value=60,   step=1,    label="Protection barrier  (% of S₀)")

        with gr.Column(scale=1):
            gr.Markdown("#### Simulation")
            n_paths = gr.Slider(1000, 50000, value=10000, step=1000, label="Number of paths")
            n_steps = gr.Slider(50,   504,   value=252,   step=1,    label="Time steps")
            gr.Markdown(
                "<br><br>",
            )
            price_btn = gr.Button("▶  Price", variant="primary", size="lg")

    gr.Markdown("---")

    # outputs
    summary_md = gr.Markdown("*Hit **Price** to run the simulation.*")

    with gr.Row():
        plot_paths = gr.Plot(label="Sample paths & barriers")

    with gr.Row():
        plot_dist  = gr.Plot(label="Payoff distribution")
        plot_hits  = gr.Plot(label="Autocall / coupon hit rates")

    # Putting everything together
    inputs = [S0, sigma, r, T, notional, coupon, autocall, coupon_b, protect, n_paths, n_steps]
    outputs = [plot_paths, plot_dist, plot_hits, summary_md]

    price_btn.click(fn=run_pricer, inputs=inputs, outputs=outputs)

    gr.Markdown(
        "<br><small style='color:#475569'>v1 · no dividends · flat vol · flat rates · "
        "no memory feature · Black-Scholes GBM</small>"
    )


if __name__ == "__main__":
    demo.launch(share=True)   # share=True prints a public gradio.live link valid for 72h