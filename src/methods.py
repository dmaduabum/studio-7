"""
Estimation Module (methods.py)

Implements three regression estimators:
  - OLS   : Ordinary Least Squares
  - LAD   : Least Absolute Deviations (Quantile Regression)
  - Huber : Huber Regression
"""

from sklearn.linear_model import LinearRegression, QuantileRegressor, HuberRegressor

def fit_model(X, y, method):
    """
    Fit one regression model.

    Parameters
    ----------
    X : ndarray, shape (n, p)
        Predictor matrix.
    y : ndarray, shape (n,)
        Response vector.
    method : str
        One of {"OLS", "LAD", "Huber"}.

    Returns
    -------
    beta_hat : ndarray
        Estimated regression coefficients.
    """
    # lower method name
    m = method.lower()

    # ----- Ordinary Least Squares -----
    if m == "ols":
        # Minimizes sum of squared residuals
        model = LinearRegression(fit_intercept=False)
        model.fit(X, y)
        return model.coef_

    # ----- Least Absolute Deviations -----
    elif m == "lad":
        # Median regression (quantile = 0.5)
        # alpha=0 removes L2 regularization
        model = QuantileRegressor(quantile=0.5, alpha=0.0,
                                  fit_intercept=False, solver="highs")
        model.fit(X, y)
        return model.coef_

    # ----- Huber Regression -----
    elif m == "huber":
        # Combines OLS (small residuals) and LAD (large residuals)
        model = HuberRegressor(fit_intercept=False, max_iter=1000)
        model.fit(X, y)
        return model.coef_

    # ----- Error handling -----
    else:
        raise ValueError("Unknown method: choose 'OLS', 'LAD', or 'Huber'")
