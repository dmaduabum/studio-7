"""
Evaluation Module (metrics.py)

Contains helper functions for computing MSE and organizing
simulation results into a tidy DataFrame.
"""

import numpy as np
import pandas as pd

def compute_mse(beta_hat, beta_true):
    """
    Compute mean squared error (MSE):
        MSE = mean((beta_hat - beta_true)^2)
    """
    return float(np.mean((beta_hat - beta_true) ** 2))

def record_result(method, params, mse):
    """
    Create a dictionary with one simulation result.

    Each result includes:
      - simulation parameters (n, p, df, rho, snr, rep)
      - estimator name (method)
      - performance metric (mse)
    """
    result = dict(params)
    result["method"] = method
    result["mse"] = mse
    return result

def results_to_dataframe(results):
    """
    Convert a list of result dictionaries into a DataFrame.
    Each row corresponds to one simulation run.
    """
    return pd.DataFrame(results)
