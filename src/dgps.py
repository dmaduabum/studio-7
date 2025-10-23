"""
Data Generation Module (dgps.py)

Simulates data from the linear model:
    y = Xβ + ε
with user-controlled:
  - design correlation (rho)
  - tail heaviness of errors (Student-t df)
  - signal-to-noise ratio (SNR)
"""

import numpy as np
from scipy import stats

def ar1_covariance(p, rho):
    """
    Construct an AR(1) covariance matrix with entries rho^{|j - k|}.
    This defines how correlated the predictors are.
    """
    indices = np.arange(p)
    return rho ** np.abs(indices[:, None] - indices[None, :])

def sample_design_matrix(n, p, rho, rng):
    """
    Generate design matrix X with AR(1) correlation structure.
    - n: number of samples
    - p: number of features
    - rho: correlation parameter
    """
    # Compute the target covariance matrix
    Sigma = ar1_covariance(p, rho)

    # Apply Cholesky decomposition to impose correlation structure
    L = np.linalg.cholesky(Sigma)

    # Draw n × p independent standard normal values
    Z = rng.standard_normal(size=(n, p))

    # Multiply by Lᵀ to introduce the correlations
    X = Z @ L.T
    return X

def sample_true_coefficients(p, rng):
    """
    Draw the true regression coefficients beta ~ N(0, I).
    """
    beta = rng.standard_normal(p)
    return beta

def compute_sigma_for_snr(X, beta, target_snr):
    """
    Compute the noise standard deviation (sigma) so that:
        Var(Xβ) / sigma² ≈ target_snr
    """
    # Compute variance of the signal (Xβ)
    signal_var = (X@ beta).T @ (X@ beta)

    # Rearrange SNR = signal_var / sigma²  →  sigma² = signal_var / SNR
    sigma2 = signal_var / float(target_snr)

    # Avoid division by zero or negative due to numerical precision
    sigma = np.sqrt(max(sigma2, 1e-12))
    return sigma

def sample_error_vector(n, df, sigma, rng):
    """
    Sample the noise vector ε.

    - If df = ∞: Gaussian noise N(0, σ²)
    - If df < ∞: Student-t noise with heavier tails
    """
    if np.isinf(df):
        # Gaussian case
        return rng.standard_normal(n) * sigma

    # Student-t case: draws have heavier tails
    t_samples = stats.t.rvs(df, size=n, random_state=rng)

    # Scale so the errors have approximately variance sigma²
    if df > 2:
        scale = sigma / np.sqrt(df / (df - 2))  # theoretical variance of t(df)
    else:
        scale = sigma / np.std(t_samples)       # fallback for df ≤ 2
    return t_samples * scale

def simulate_dataset(n, gamma, rho, df, snr, seed=None):
    """
    Simulate one dataset (X, y, beta) with given parameters.
    """
    # Create reproducible random generator
    rng = np.random.default_rng(seed)

    # Compute number of predictors from gamma = p / n
    p = int(round(gamma * n))

    # Generate correlated design matrix
    X = sample_design_matrix(n, p, rho, rng)

    # Generate true coefficients
    beta = sample_true_coefficients(p, rng)

    # Compute noise scale for target SNR
    sigma = compute_sigma_for_snr(X, beta, snr)

    # Generate heavy-tailed noise
    errors = sample_error_vector(n, df, sigma, rng)

    # Response variable
    y = X @ beta + errors

    # Package results
    return {
        "X": X,
        "y": y,
        "beta": beta,
        "sigma": sigma,
        "params": {"n": n, "p": p, "gamma": gamma, "rho": rho,
                   "df": df, "snr": snr, "seed": seed}
    }


