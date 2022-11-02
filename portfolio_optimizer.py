import pandas as pd
import numpy as np
from scipy.optimize import minimize

class Portfolio:
    """
    Portfolio

    ...

    Attributes
    ----------
    price_df : DataFrame
        a DataFrame of asset prices, names of assets as columns and time index as index
    returns : DataFrame
        a DataFrame of price returns, names of assets as columns and time index as index
    mean_returns : float
        the mean returns of each asset
    covariance : ndarray
        the covariance matrix of returns of assets

    Methods
    -------
    compute_portfolio_return(weights)
        computes portfolio return
    compute_portfolio_variance(weights)
        computes portfolio variance
    minimize_func(func)
        minimizes objective function by adjusting portfolio weights
    """
    
    def __init__(self, price_df, apply_shrinkage = False):
        self.price_df = price_df
        self.returns = (self.price_df.shift(-1) / self.price_df - 1).dropna()
        self.mean_returns = self.returns.mean()
        self.covariance = np.cov(self.returns.values.transpose())
        if apply_shrinkage:
            self.covariance = np.identity(len(self.price_df.columns)) * self.covariance
        self.res = ''
        
    def compute_portfolio_return(self, weights):
        return np.dot(self.mean_returns, weights)
    
    def compute_portfolio_variance(self, weights):
        return np.dot(np.dot(weights, self.covariance), weights)
    
    def minimize_func(self, func):
        x0 = np.full(len(self.price_df.columns), 1/len(self.price_df.columns))
        eq_cons = {'type': 'eq',
                   'fun' : lambda x: np.array([1 - sum(x)]),
                   'jac' : lambda x: np.array([-1]*len(x))}
        bounds = [[0,1]]*len(x0)
        res = minimize(func, x0, method='SLSQP',
                       constraints = eq_cons, bounds=bounds,
                       options={'ftol': 1e-100})
        return res