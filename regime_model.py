"""
Monte Carlo regime-switching simulation, fitted to a real stock.

Step 2 of the build: the user types a ticker, we download its daily price
history, fit a 2-state Hidden Markov Model (Calm / Stressed) to its daily
returns, and then run the same Monte Carlo + buy-and-hold test as before,
but with that stock's own regime means, vols and transition matrix.

Requires:  pip install -r requirements.txt
Run:       python regime_model.py FTAI
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# 0. Settings
# ---------------------------------------------------------------
REGIME_NAMES = ["Calm", "Stressed"]   # index 0 = Calm, index 1 = Stressed

HISTORY_PERIOD = "10y"       # how much history to fit on ("5y", "max", ...)
TRADING_DAYS_PER_YEAR = 252  # markets are closed weekends/holidays
N_DAYS = TRADING_DAYS_PER_YEAR * 1   # change the multiplier for multi-year runs
N_PATHS = 1000
SEED = 42                    # fixed seed = reproducible runs; set to None for fresh randomness

rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------
# 1. Get historical data for a ticker
# ---------------------------------------------------------------
def download_prices(ticker, period=HISTORY_PERIOD):
    """Return a pandas Series of daily adjusted closing prices."""
    import yfinance as yf  # imported here so the rest of the file works without it

    data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if data is None or data.empty:
        raise ValueError(f"No data found for ticker '{ticker}'. Check the symbol.")

    close = data["Close"]
    # newer yfinance versions return a one-column DataFrame; squeeze it to a Series
    if hasattr(close, "columns"):
        close = close.iloc[:, 0]
    close = close.dropna()

    if len(close) < 300:
        raise ValueError(f"Only {len(close)} days of data for '{ticker}' -- "
                         "need at least ~300 to fit regimes reliably.")
    return close


# ---------------------------------------------------------------
# 2. Fit a 2-state Gaussian Hidden Markov Model (Baum-Welch / EM)
# ---------------------------------------------------------------
# We only observe returns, not regimes. The HMM assumes each day's return
# comes from Normal(mean[s], vol[s]) where s is a hidden regime that follows
# a Markov chain. EM alternates between:
#   E-step: given current parameters, how likely was each day to be Calm/Stressed?
#   M-step: given those likelihoods, re-estimate means, vols and transitions.
# It repeats until the parameters stop changing.

def _normal_pdf(x, mean, vol):
    return np.exp(-0.5 * ((x - mean) / vol) ** 2) / (vol * np.sqrt(2 * np.pi))


def _forward_backward(returns, mean, vol, trans, start_prob):
    """Scaled forward-backward pass. Returns per-day regime probabilities
    (gamma), expected transition counts (xi_sum) and the log-likelihood."""
    T, K = len(returns), len(mean)
    emis = np.column_stack([_normal_pdf(returns, mean[k], vol[k]) for k in range(K)])
    emis = np.maximum(emis, 1e-300)  # avoid exact zeros on extreme days

    alpha = np.zeros((T, K))
    scale = np.zeros(T)
    alpha[0] = start_prob * emis[0]
    scale[0] = alpha[0].sum()
    alpha[0] /= scale[0]
    for t in range(1, T):
        alpha[t] = (alpha[t - 1] @ trans) * emis[t]
        scale[t] = alpha[t].sum()
        alpha[t] /= scale[t]

    beta = np.zeros((T, K))
    beta[-1] = 1.0
    for t in range(T - 2, -1, -1):
        beta[t] = (trans @ (emis[t + 1] * beta[t + 1])) / scale[t + 1]

    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)

    xi_sum = np.zeros((K, K))
    for t in range(T - 1):
        xi = alpha[t][:, None] * trans * (emis[t + 1] * beta[t + 1])[None, :]
        xi_sum += xi / xi.sum()

    return gamma, xi_sum, np.log(scale).sum()


def fit_regimes(returns, n_iter=200, tol=1e-6):
    """Fit Calm/Stressed regimes to an array of daily returns.
    Returns (mean, vol, transition_matrix, gamma) with state 0 = Calm."""
    returns = np.asarray(returns, dtype=float)

    # Starting guess: split days by size of move. The quietest 70% seed
    # "Calm", the loudest 30% seed "Stressed".
    cutoff = np.quantile(np.abs(returns - returns.mean()), 0.7)
    loud = np.abs(returns - returns.mean()) > cutoff
    mean = np.array([returns[~loud].mean(), returns[loud].mean()])
    vol = np.array([returns[~loud].std(), returns[loud].std()])
    trans = np.array([[0.95, 0.05], [0.10, 0.90]])
    start_prob = np.array([0.5, 0.5])

    prev_ll = -np.inf
    for _ in range(n_iter):
        gamma, xi_sum, ll = _forward_backward(returns, mean, vol, trans, start_prob)

        # M-step
        weights = gamma.sum(axis=0)
        mean = (gamma * returns[:, None]).sum(axis=0) / weights
        vol = np.sqrt((gamma * (returns[:, None] - mean) ** 2).sum(axis=0) / weights)
        vol = np.maximum(vol, 1e-6)
        trans = xi_sum / xi_sum.sum(axis=1, keepdims=True)
        start_prob = gamma[0]

        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

    # Make sure index 0 is the low-volatility (Calm) state
    if vol[0] > vol[1]:
        order = [1, 0]
        mean, vol = mean[order], vol[order]
        trans = trans[np.ix_(order, order)]
        gamma = gamma[:, order]

    return mean, vol, trans, gamma


# ---------------------------------------------------------------
# 3. Single-path generator (regime path + price path)
# ---------------------------------------------------------------
def simulate_path(n_days, starting_regime, starting_price, mean, vol, trans):
    regimes = np.zeros(n_days, dtype=int)
    returns = np.zeros(n_days)

    current_regime = starting_regime
    for t in range(n_days):
        current_regime = rng.choice(2, p=trans[current_regime])
        regimes[t] = current_regime
        returns[t] = rng.normal(mean[current_regime], vol[current_regime])

    prices = starting_price * np.cumprod(1 + returns)
    return regimes, prices


# ---------------------------------------------------------------
# 4. Strategy: buy-and-hold
# ---------------------------------------------------------------
def buy_and_hold_return(prices, starting_price):
    return (prices[-1] - starting_price) / starting_price


# ---------------------------------------------------------------
# 5. Monte Carlo loop
# ---------------------------------------------------------------
def run_monte_carlo(n_paths, n_days, starting_regime, starting_price,
                    mean, vol, trans, n_sample_paths_to_keep=15):
    final_returns = np.zeros(n_paths)
    pct_time_stressed = np.zeros(n_paths)
    sample_price_paths, sample_regime_paths = [], []

    for i in range(n_paths):
        regimes, prices = simulate_path(n_days, starting_regime, starting_price,
                                        mean, vol, trans)
        final_returns[i] = buy_and_hold_return(prices, starting_price)
        pct_time_stressed[i] = np.mean(regimes == 1)

        if i < n_sample_paths_to_keep:
            sample_price_paths.append(prices)
            sample_regime_paths.append(regimes)

    return final_returns, pct_time_stressed, sample_price_paths, sample_regime_paths


# ---------------------------------------------------------------
# 6. Plotting
# ---------------------------------------------------------------
def plot_history(ticker, prices, stressed_prob):
    """Real price history with the days the model calls Stressed shaded red."""
    fig, ax = plt.subplots(figsize=(13, 4.5))
    dates = prices.index[1:]            # returns start one day after prices
    p = prices.values[1:]
    ax.plot(dates, p, color="black", linewidth=1)
    ax.fill_between(dates, p.min(), p.max(), where=stressed_prob > 0.5,
                    color="red", alpha=0.15, step="mid", label="Fitted Stressed regime")
    ax.set_title(f"{ticker}: historical prices with fitted regimes")
    ax.set_ylabel("Price")
    ax.legend(loc="upper left")
    plt.tight_layout()
    fname = f"{ticker}_historical_regimes.png"
    plt.savefig(fname, dpi=150)
    print(f"Saved plot to {fname}")


def plot_results(ticker, final_returns, pct_time_stressed,
                 sample_price_paths, sample_regime_paths):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    for prices in sample_price_paths:
        ax.plot(prices, linewidth=1, alpha=0.8)
    ax.set_title(f"{ticker}: sample simulated price paths")
    ax.set_xlabel("Day")
    ax.set_ylabel("Price")

    ax = axes[0, 1]
    prices0, regimes0 = sample_price_paths[0], sample_regime_paths[0]
    ax.plot(prices0, color="black", linewidth=1.2)
    ax.fill_between(np.arange(len(prices0)), prices0.min(), prices0.max(),
                    where=regimes0 == 1, color="red", alpha=0.15, step="mid",
                    label="Stressed regime")
    ax.set_title("One simulated path: red = Stressed regime periods")
    ax.set_xlabel("Day")
    ax.set_ylabel("Price")
    ax.legend()

    ax = axes[1, 0]
    ax.hist(final_returns, bins=40, color="steelblue", edgecolor="white")
    ax.axvline(np.median(final_returns), color="black", linestyle="--", label="Median")
    ax.axvline(np.percentile(final_returns, 5), color="red", linestyle="--", label="5th pct")
    ax.set_title(f"Distribution of buy-and-hold final returns\n({len(final_returns)} simulated paths)")
    ax.set_xlabel("Final return")
    ax.set_ylabel("Number of paths")
    ax.legend()

    ax = axes[1, 1]
    ax.scatter(pct_time_stressed, final_returns, s=8, alpha=0.4, color="darkorange")
    ax.set_title("Final return vs. time spent in Stressed regime")
    ax.set_xlabel("% of days in Stressed regime")
    ax.set_ylabel("Final return")

    plt.tight_layout()
    fname = f"{ticker}_sim_results.png"
    plt.savefig(fname, dpi=150)
    print(f"Saved plot to {fname}")


def print_fitted_params(ticker, mean, vol, trans, n_days_hist):
    p_stress = trans[0, 1] / (trans[0, 1] + trans[1, 0])  # long-run share Stressed
    print(f"\nFitted regimes for {ticker} ({n_days_hist} trading days of history)")
    for k, name in enumerate(REGIME_NAMES):
        print(f"  {name:<9} mean {mean[k]:+.3%}/day   vol {vol[k]:.2%}/day   "
              f"avg spell {1 / (1 - trans[k, k]):.0f} days")
    print("  Transition matrix (rows = from, cols = to):")
    print(f"    Calm     -> Calm {trans[0, 0]:.3f}   Stressed {trans[0, 1]:.3f}")
    print(f"    Stressed -> Calm {trans[1, 0]:.3f}   Stressed {trans[1, 1]:.3f}")
    print(f"  Long-run share of days Stressed: {p_stress:.1%}\n")


# ---------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------
if __name__ == "__main__":
    import sys
    ticker = (sys.argv[1] if len(sys.argv) > 1
              else input("Enter a stock ticker (e.g. AAPL, SPY, TSLA): ")).strip().upper()

    hist_prices = download_prices(ticker)
    hist_returns = hist_prices.pct_change().dropna().values

    mean, vol, trans, gamma = fit_regimes(hist_returns)
    print_fitted_params(ticker, mean, vol, trans, len(hist_returns))

    # Start the simulation from today's real price and today's fitted regime
    starting_price = float(hist_prices.iloc[-1])
    starting_regime = int(gamma[-1, 1] > 0.5)
    print(f"Starting price ${starting_price:.2f}, current regime: "
          f"{REGIME_NAMES[starting_regime]} (P(Stressed) = {gamma[-1, 1]:.0%})\n")

    final_returns, pct_time_stressed, sample_price_paths, sample_regime_paths = \
        run_monte_carlo(N_PATHS, N_DAYS, starting_regime, starting_price, mean, vol, trans)

    print(f"Simulated {N_PATHS} paths of {N_DAYS} days each, buy-and-hold strategy\n")
    print(f"Mean final return:      {np.mean(final_returns):+.2%}")
    print(f"Median final return:    {np.median(final_returns):+.2%}")
    print(f"Std dev of return:      {np.std(final_returns):.2%}")
    print(f"Worst 5% of paths:      {np.percentile(final_returns, 5):+.2%}")
    print(f"Best 5% of paths:       {np.percentile(final_returns, 95):+.2%}")
    print(f"Fraction of paths lost money: {np.mean(final_returns < 0):.1%}")
    print(f"\nAvg % of time spent in Stressed regime: {np.mean(pct_time_stressed):.1%}")
    corr = np.corrcoef(pct_time_stressed, final_returns)[0, 1]
    print(f"Correlation(time in Stressed, final return): {corr:.2f}\n")

    plot_history(ticker, hist_prices, gamma[:, 1])
    plot_results(ticker, final_returns, pct_time_stressed,
                 sample_price_paths, sample_regime_paths)
