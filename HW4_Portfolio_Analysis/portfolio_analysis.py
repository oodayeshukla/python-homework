#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 10:02:42 2021

@author: oodaye
"""

#%% Homework 4

#%% 
import pandas as pd
import numpy as np
import matplotlib as plt
import datetime as dt
from pathlib import Path
import os

#%% set up the directory
os.chdir("/home/oodaye/Fintech/class_repo/python-homework-repo/python-homework")

#%% ingest the data 
basedir = '/home/oodaye/Fintech/class_repo/python-homework-repo/python-homework/HW4_Portfolio_Analysis'

aapl_pth = Path(basedir + "/Resources/aapl_historical.csv")
algo_pth = Path(basedir + "/Resources/algo_returns.csv")
cost_pth = Path(basedir + "/Resources/cost_historical.csv")
goog_pth = Path(basedir + "/Resources/goog_historical.csv")
sp500_pth = Path(basedir + "/Resources/sp500_history.csv")
whale_pth = Path(basedir + "/Resources/whale_returns.csv")

aapl_df = pd.read_csv(aapl_pth, index_col=("Trade DATE"))
algo_df = pd.read_csv(algo_pth, index_col=("Date"))
cost_df = pd.read_csv(cost_pth, index_col=("Trade DATE"))
goog_df = pd.read_csv(goog_pth, index_col=("Trade DATE"))
sp500_df = pd.read_csv(sp500_pth, index_col=("Date"))
whale_df = pd.read_csv(whale_pth, index_col=("Date"))

#%% clean the data -- 
## Count nulls 
aapl_nulls = aapl_df.isnull().sum()

## Drop the nulls 

#%%  Fro SP500 data 

# Reading S&P 500 Closing Prices
# Check Data Types
# Fix Data Types
# Calculate Daily Returns
# Drop nulls
# Rename `Close` Column to be specific to this portfolio.
  
  
#%% # Join Whale Returns, Algorithmic Returns, and the S&P 500 Returns into a single DataFrame with columns for each portfolio's returns.


#%% # Plot daily returns of all portfolios


#%%  cumulative returns of all portfolios 

 # Calculate cumulative returns of all portfolios

# Plot cumulative returns


#%%

# Risk Analysis
# Determine the risk of each portfolio:
# Create a box plot for each portfolio. 
# Calculate the standard deviation for all portfolios
# Determine which portfolios are riskier than the S&P 500
# Calculate the Annualized Standard Deviation




#%% evaluate data 
## returns are pct change -- hence the first row is NaN


