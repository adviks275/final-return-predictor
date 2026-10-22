"""
Check that the Baum-Welch fitter recovers known parameters.

Generates ~10 years of daily returns from a 2-regime model whose true
parameters we choose (close to what the model fitted for FTAI), fits the HMM
to those returns, and compares fitted vs. true values. Repeats over several
random seeds so one lucky draw can't make the fitter look good.

Run:  python test_fitter.py
"""
import numpy as np

from regime_model import fit_regimes

TRUE_MEAN = np.array([0.0014, 0.0029])
TRUE_VOL = np.array([0.016, 0.055])
TRUE_TRANS = np.array([[0.967, 0.033],
                       [0.080, 0.920]])
N_DAYS = 2513          # same length as the FTAI history
SEEDS = range(10)


def generate(seed):
    rng = np.random.default_rng(seed)
    regimes = np.zeros(N_DAYS, dtype=int)
    for t in range(1, N_DAYS):
        regimes[t] = rng.choice(2, p=TRUE_TRANS[regimes[t - 1]])
    returns = rng.normal(TRUE_MEAN[regimes], TRUE_VOL[regimes])
    return returns, regimes


def main():
    rows, accuracy = [], []
    for seed in SEEDS:
        returns, regimes = generate(seed)
        mean, vol, trans, gamma = fit_regimes(returns)
        rows.append(np.concatenate([mean, vol, [trans[0, 1], trans[1, 0]]]))
        accuracy.append(np.mean((gamma[:, 1] > 0.5) == (regimes == 1)))
    rows = np.array(rows)
    truth = np.concatenate([TRUE_MEAN, TRUE_VOL, [TRUE_TRANS[0, 1], TRUE_TRANS[1, 0]]])
    names = ["Calm mean", "Stressed mean", "Calm vol", "Stressed vol",
             "P(Calm -> Stressed)", "P(Stressed -> Calm)"]

    print(f"Fitter recovery over {len(SEEDS)} simulated {N_DAYS}-day histories\n")
    print(f"  {'parameter':<20} {'true':>9} {'fitted (avg)':>13} {'spread (sd)':>12}")
    for name, t, col in zip(names, truth, rows.T):
        print(f"  {name:<20} {t:>9.4f} {col.mean():>13.4f} {col.std():>12.4f}")
    print(f"\nDays assigned to the correct regime: {np.mean(accuracy):.1%} on average "
          f"(worst seed {np.min(accuracy):.1%})")

    # Vols and transitions should come back close; means are noisy by nature.
    assert np.allclose(rows[:, 2:4].mean(axis=0), TRUE_VOL, rtol=0.10), "vols not recovered"
    assert np.mean(accuracy) > 0.85, "regime classification too weak"
    print("\nPASS: volatilities within 10% of truth and regimes classified correctly")


if __name__ == "__main__":
    main()
