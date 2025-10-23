"""
Visualization Module (visualize.py)

Generates publication-quality figures summarizing simulation results.

"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set consistent plot style
sns.set_theme(context="notebook", style="whitegrid", font_scale=1.2)
palette = {"OLS": "#1f77b4", "LAD": "#2ca02c", "Huber": "#d62728"}

def plot_mse_vs_df(df, out_dir="results/figures", width=5, height=3):
    """
    Create MSE vs df plots for each combination of SNR and gamma.

    Parameters
    ----------
    df : pandas.DataFrame
        Simulation results with columns:
        ["method", "n", "p", "gamma", "rho", "df", "snr", "rep", "mse"]
    out_dir : str
        Directory where figures will be saved.
    width, height : float
        Figure size in inches.
    """

    # Compute mean MSE grouped by method and parameters
    summary = (
        df.groupby(["method", "gamma", "snr", "df"], as_index=False)["mse"]
          .mean()
    )

    # For each combination of (gamma, snr), make a small multiple
    gammas = sorted(df["gamma"].unique())
    snrs = sorted(df["snr"].unique())

    for gamma in gammas:
        for snr in snrs:
            subset = summary[(summary["gamma"] == gamma) & (summary["snr"] == snr)]
            plt.figure(figsize=(width, height))

            # Lineplot: MSE vs df, one line per method
            sns.lineplot(
                data=subset,
                x="df",
                y="mse",
                hue="method",
                marker="o",
                palette=palette,
            )

            # Make labels readable and clear
            plt.title(f"MSE vs Tail Heaviness (γ={gamma}, SNR={snr})")
            plt.xlabel("Degrees of Freedom (df)")
            plt.ylabel("Mean Squared Error (MSE)")
            plt.legend(title="Estimator", frameon=False)
            plt.tight_layout()

            # Save figure
            filename = f"mse_vs_df_gamma{gamma}_snr{snr}.png"
            out_path = os.path.join(out_dir, filename)
            plt.savefig(out_path, dpi=300)
            plt.close()
            print(f"Saved figure: {out_path}")


def main(csv_path="results/raw/simulation_results.csv", out_dir="results/figures"):
    """
    Load simulation results and create all figures.
    """
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} simulation rows from {csv_path}")

    # Generate MSE vs df plots for all gamma × SNR combinations
    plot_mse_vs_df(df, out_dir=out_dir)


if __name__ == "__main__":
    # Example usage: python -m src.visualize
    main()
