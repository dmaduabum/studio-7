"""
Simulation Driver (simulation.py)

Runs the full Monte Carlo experiment:
  1. Generate data using simulate_dataset()
  2. Fit OLS, LAD, and Huber using fit_model()
  3. Compute MSE for each method
  4. Save all results in a tidy CSV file
"""

import os
import numpy as np
from src.dgps import simulate_dataset
from src.methods import fit_model
from src.metrics import compute_mse, record_result, results_to_dataframe

def run_simulation(
    n=200,
    gammas=(0.2, 0.5, 0.8),       # aspect ratios (p/n)
    rhos=(0.1, 0.5, 0.9),         # positive corr among predictors
    dfs=(1, 2, 3, 20, np.inf),    # Student-t degrees of freedom
    snrs=(1, 5, 10),              # signal-to-noise ratios
    reps=5,                       # number of replicates
    seed=123,
    out_path="results/raw/simulation_results.csv"
):

    """Run simulation grid and save results as CSV."""
    # Random number generator for reproducibility
    rng = np.random.default_rng(seed)

    # Store all results here
    all_results = []

    # Estimators to compare
    methods = ["OLS", "LAD", "Huber"]

    # Loop over experimental settings
    for gamma in gammas:
        p = int(round(gamma * n))  # compute p from gamma
        for rho in rhos:
            for df in dfs:
                for snr in snrs:
                    for rep in range(1, reps + 1):

                        # Generate one dataset with its own random seed
                        data = simulate_dataset(n, gamma, rho, df, snr,
                                                seed=rng.integers(1e9))
                        X, y, beta_true = data["X"], data["y"], data["beta"]

                        # Record simulation parameters
                        params = {
                            "n": n,
                            "p": p,
                            "gamma": gamma,
                            "rho": rho,
                            "df": df,
                            "snr": snr,
                            "rep": rep,
                        }

                        # Fit each regression method and compute MSE
                        for method in methods:
                            beta_hat = fit_model(X, y, method)
                            mse = compute_mse(beta_hat, beta_true)
                            result_row = record_result(method, params, mse)
                            all_results.append(result_row)

    # Convert results to DataFrame
    df = results_to_dataframe(all_results)

    # Save results to CSV
    df.to_csv(out_path, index=False)
    print(f"Saved results to {out_path} ({len(df)} rows)")
    return df


if __name__ == "__main__":
    # Example small run for quick testing
    df = run_simulation(reps=3)
    print(df.head())
