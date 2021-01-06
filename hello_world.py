#!/usr/bin/env python3.7

import pandas as pd
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import objective_functions

vtsax_raw_history = pd.read_csv("../stock-data/VTSAX.csv", parse_dates=True, index_col="Date")
vtiax_raw_history = pd.read_csv("../stock-data/VTIAX.csv", parse_dates=True, index_col="Date")
vbtlx_raw_history = pd.read_csv("../stock-data/VBTLX.csv", parse_dates=True, index_col="Date")
vtabx_raw_history = pd.read_csv("../stock-data/VTABX.csv", parse_dates=True, index_col="Date")

vtsax_history = vtsax_raw_history[['Adj Close']].copy().rename(columns={"Adj Close": "VTSAX"})
vtiax_history = vtiax_raw_history[['Adj Close']].copy().rename(columns={"Adj Close": "VTIAX"})
vbtlx_history = vbtlx_raw_history[['Adj Close']].copy().rename(columns={"Adj Close": "VBTLX"})
vtabx_history = vtabx_raw_history[['Adj Close']].copy().rename(columns={"Adj Close": "VTABX"})

merged_history = pd.merge(vtsax_history, vtiax_history, left_index=True, right_index=True)
merged_history = merged_history.merge(vbtlx_history, left_index=True, right_index=True)
merged_history = merged_history.merge(vtabx_history, left_index=True, right_index=True)
print(merged_history)

# Calculate expected returns and sample covariance
mu = expected_returns.mean_historical_return(merged_history)
print(mu)
S = risk_models.sample_cov(merged_history)
print(S)

# Optimise for 10% return, maybe reasonable for long term goals
ef = EfficientFrontier(mu, S)
ef.add_objective(objective_functions.L2_reg, gamma=1)
raw_weights = ef.max_sharpe()
#raw_weights = ef.efficient_return(target_return=0.12)
cleaned_weights = ef.clean_weights()
print(cleaned_weights)
ef.portfolio_performance(verbose=True)
