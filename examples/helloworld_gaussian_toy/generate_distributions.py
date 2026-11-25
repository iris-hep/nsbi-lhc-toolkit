import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import _BASIS_COLORS, lagrange_weights

parser = argparse.ArgumentParser()
parser.add_argument("--n_bkg", type=int, default=2_000_000)
parser.add_argument("--n_sig", type=int, default=1_000_000)
parser.add_argument("--nodes", type=float, nargs="+", default=[0, 5, 10])
args = parser.parse_args()

np.random.seed(42)

n_bkg = args.n_bkg
n_sig = args.n_sig
nodes = args.nodes

features = ["x1", "x2", "x3", "x4", "x5"]

# Background: broad distributions overlapping with signal region
background = pd.DataFrame(
    {
        "x1": np.random.normal(loc=2.8, scale=1.5, size=n_bkg),
        "x2": np.random.normal(loc=1.5, scale=1.4, size=n_bkg),
        "x3": np.random.normal(loc=3.5, scale=1.5, size=n_bkg),
        "x4": np.random.normal(loc=1.0, scale=1.4, size=n_bkg),
        "x5": np.random.normal(loc=2.5, scale=1.5, size=n_bkg),
        "fold": [np.random.choice([0, 1]) for _ in range(n_bkg)],
    }
)
background["label"] = 0
background["weight"] = 1.0 * (1_000_000 / n_bkg)  # preserves λ_b = 1e6

# data: broad distributions overlapping with signal region
data = pd.DataFrame(
    {
        "x1": np.random.normal(loc=2.8, scale=1.5, size=n_bkg),
        "x2": np.random.normal(loc=1.5, scale=1.4, size=n_bkg),
        "x3": np.random.normal(loc=3.5, scale=1.5, size=n_bkg),
        "x4": np.random.normal(loc=1.0, scale=1.4, size=n_bkg),
        "x5": np.random.normal(loc=2.5, scale=1.5, size=n_bkg),
        "fold": [np.random.choice([0, 1]) for _ in range(n_bkg)],
    }
)
data["label"] = 0

# Signal parameters indexed by node order.
signal_params = [
    {
        "x1": (4.7, 1.0),
        "x2": (3.1, 0.9),
        "x3": (3.7, 1.5),
        "x4": (1.2, 1.4),
        "x5": (2.7, 1.4),
    },
    {
        "x1": (3.7, 1.3),
        "x2": (2.3, 1.2),
        "x3": (3.7, 1.4),
        "x4": (2.5, 1.0),
        "x5": (2.7, 1.5),
    },
    {
        "x1": (3.0, 1.5),
        "x2": (1.7, 1.4),
        "x3": (5.1, 0.9),
        "x4": (1.2, 1.3),
        "x5": (4.2, 0.9),
    },
]

# Bins
os.makedirs("dataframes", exist_ok=True)
background.to_parquet("dataframes/background.parquet", index=False)
data.to_parquet("dataframes/data.parquet", index=False)

# Build basis signals keyed by node value
basis_signals = []  # list of (name, node, df)
for node, params in zip(nodes, signal_params):
    name = f"signal_{node:g}"
    df = pd.DataFrame(
        {
            feat: np.random.normal(params[feat][0], params[feat][1], n_sig)
            for feat in features
        }
    )
    df["label"] = 1
    df["weight"] = 0.01 * (1 + node) / 10 * (100_000 / n_sig)  # preserves Σw at the original n_sig=100k baseline
    df["fold"] = [np.random.choice([0, 1]) for _ in range(df.shape[0])]
    basis_signals.append((name, node, df))
    df.to_parquet(f"dataframes/{name}.parquet", index=False)

basis_dfs = [df for _, _, df in basis_signals]


def morphed_signal_df(v, basis_dfs, nodes):
    """Build a morphed signal DataFrame by concatenating basis signals with Lagrange weights."""
    weights = lagrange_weights(v, nodes)
    dfs = []
    for w, df in zip(weights, basis_dfs):
        df_copy = df.copy()
        df_copy["weight"] = w
        dfs.append(df_copy)
    return pd.concat(dfs, ignore_index=True)


validation_values = list(nodes)  # should exactly recover basis signals at nodes
new_values = [-1, 1, 7]

os.makedirs("plots", exist_ok=True)

# --- Plot 1: basis signals only (no morphing) ---
for feat in features:
    fig, ax = plt.subplots(figsize=(7, 5))

    lo = min(
        background[feat].quantile(0.01),
        *(df[feat].quantile(0.01) for _, _, df in basis_signals),
    )
    hi = max(
        background[feat].quantile(0.99),
        *(df[feat].quantile(0.99) for _, _, df in basis_signals),
    )
    bins = np.linspace(lo, hi, 51)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else 1.0

    ax.errorbar(
        bin_centers,
        np.histogram(data[feat], bins=bins, density=True)[0],
        yerr=0,
        fmt="k.",
        capsize=3,
        ms=4,
        lw=1.2,
        xerr=None,
        label="Data",
        zorder=5,
    )
    ax.hist(
        background[feat],
        bins=bins,
        density=True,
        histtype="stepfilled",
        lw=2,
        color="black",
        alpha=0.15,
        label="Background",
    )
    for (name, node, df), color in zip(basis_signals, _BASIS_COLORS):
        ax.hist(
            df[feat],
            bins=bins,
            density=True,
            histtype="step",
            lw=2,
            color=color,
            label=f"{name} (v={node:g})",
        )

    ax.set_xlabel(feat, loc="right")
    ax.set_ylabel("Density", loc="top")
    ax.legend(fontsize=8)
    fig.savefig(f"plots/{feat}.pdf", bbox_inches="tight")
    plt.close(fig)

# --- Plot 2: basis signals + validation morphs + new morphs ---
for feat in features:
    fig, ax = plt.subplots(figsize=(7, 5))

    lo = min(
        background[feat].quantile(0.01),
        *(df[feat].quantile(0.01) for _, _, df in basis_signals),
    )
    hi = max(
        background[feat].quantile(0.99),
        *(df[feat].quantile(0.99) for _, _, df in basis_signals),
    )
    bins = np.linspace(lo, hi, 51)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else 1.0

    ax.errorbar(
        bin_centers,
        np.histogram(data[feat], bins=bins, density=True)[0],
        yerr=0,
        fmt="k.",
        capsize=3,
        ms=4,
        lw=1.2,
        xerr=None,
        label="Data",
        zorder=5,
    )
    ax.hist(
        background[feat],
        bins=bins,
        density=True,
        histtype="stepfilled",
        lw=2,
        color="black",
        alpha=0.15,
        label="Background",
    )
    for (name, node, df), color in zip(basis_signals, _BASIS_COLORS):
        ax.hist(
            df[feat],
            bins=bins,
            density=True,
            histtype="step",
            lw=2,
            ls="--",
            color=color,
            label=f"{name} (v={node:g})",
        )

    for v in new_values + validation_values:
        morphed = morphed_signal_df(v, basis_dfs, nodes)
        h, _ = np.histogram(
            morphed[feat], bins=bins, weights=morphed["weight"], density=True
        )
        ax.stairs(h, bins, color="black", ls="--", lw=1, label=f"Morphed v={v:g}")

    ax.set_xlabel(feat)
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    fig.savefig(f"plots/{feat}_morphed.pdf", bbox_inches="tight")
    plt.close(fig)

# --- Plot 3: signal cross sections (sum of weights) vs v ---
v_scan = np.linspace(nodes[0], nodes[-1], 200)
basis_xsecs = [df["weight"].sum() for _, _, df in basis_signals]
morphed_xsecs = []
for v in v_scan:
    lw = lagrange_weights(v, nodes)
    xsec = sum(w * xs for w, xs in zip(lw, basis_xsecs))
    morphed_xsecs.append(xsec)

fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(v_scan, morphed_xsecs, lw=2, color="black", label="Morphed")
for (name, node, _), xs, color in zip(basis_signals, basis_xsecs, _BASIS_COLORS):
    ax.scatter([node], [xs], color=color, zorder=5, s=60, label=name)
ax.set_xlabel("Signal parameter v", loc="right")
ax.set_ylabel("Cross section (sum of weights)", loc="top")
ax.legend()
fig.savefig("plots/xsec_vs_v.pdf", bbox_inches="tight")
plt.close(fig)

print("Done. Plots saved to plots/")
