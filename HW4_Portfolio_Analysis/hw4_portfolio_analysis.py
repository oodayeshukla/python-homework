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
import matplotlib.pyplot as plt
import datetime as dt
from pathlib import Path
import seaborn as sns
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
cost_df = pd.read_csv(cost_pth, index_col=("Trade DATE"))
goog_df = pd.read_csv(goog_pth, index_col=("Trade DATE"))

## handle these data separately to align the date format for later concatenation
sp500_df = pd.read_csv(sp500_pth)
sp500_df["Date"] = pd.to_datetime(sp500_df["Date"])
sp500_df = sp500_df.set_index("Date")
sp500_df = sp500_df.rename(columns = {'Close':'SP500'})

algo_df = pd.read_csv(algo_pth)
algo_df["Date"] = pd.to_datetime(algo_df["Date"])
algo_df = algo_df.set_index("Date")

whale_df = pd.read_csv(whale_pth)
whale_df["Date"] = pd.to_datetime(whale_df["Date"])
whale_df = whale_df.set_index("Date")


#%% clean the data -- 
## Count nulls 
aapl_nulls = aapl_df.isnull().sum()
algo_nulls = algo_df.isnull().sum()
cost_nulls = cost_df.isnull().sum()
goog_nulls = goog_df.isnull().sum()
sp500_nulls = sp500_df.isnull().sum()
whale_nulls = whale_df.isnull().sum()

print(f"Nulls: AAPL={aapl_nulls}, Algo={algo_nulls}, Cost={cost_nulls}, GOOG={goog_nulls}, SP500={sp500_nulls}, Whale={whale_nulls}")

#%% convert the data type in SP500 to numeric from string 
## source: https://pbpython.com/currency-cleanup.html

sp500_df['SP500'] = sp500_df['SP500'].replace({'\$': '', ',': ''}, regex=True).astype(float)
sp500_df['SP500'] = sp500_df['SP500'].pct_change()     ## compute the returns for SP500 

#%% Drop the nulls 
aapl = aapl_df.dropna()
algo = algo_df.dropna()
cost = cost_df.dropna()
goog = goog_df.dropna()
sp500 = sp500_df.dropna()
whale = whale_df.dropna()

### the result is cleaned data 

### sort the data to with increasing dates 



#%%  Fro SP500 data 

# Reading S&P 500 Closing Prices
# Check Data Types
# Fix Data Types
# Calculate Daily Returns
# Drop nulls
# Rename `Close` Column to be specific to this portfolio.
  
#%% Join Whale Returns, Algorithmic Returns, and the S&P 500 Returns into a single DataFrame with columns for each portfolio's returns.

jnd_ret = pd.concat([whale, algo, sp500], axis="columns", join = "inner")
jnd_ret = jnd_ret.sort_index(ascending=False)

#%% # Plot daily returns of all portfolios

plt1 = jnd_ret.plot(ylabel = "Daily Returns", xlabel="Date")

## use the matplotlib subplots 


fig = plt.figure(figsize=[16,15])   ## create a figure 

fig.autofmt_xdate(rotation=90) 
fig.add_subplot(2,2,1)
jnd_ret[jnd_ret.columns[0]].plot(xlabel="Date", ylabel="Returns", title=jnd_ret.columns[0].strip())

fig.add_subplot(2,2,2) 
jnd_ret[jnd_ret.columns[1]].plot(xlabel="Date", ylabel="Returns", title=jnd_ret.columns[1].strip())

fig.add_subplot(2,2,3) 
jnd_ret[jnd_ret.columns[2]].plot(xlabel="Date", ylabel="Returns", title=jnd_ret.columns[2].strip())

fig.add_subplot(2,2,4) 
jnd_ret[jnd_ret.columns[3]].plot(xlabel="Date", ylabel="Returns", title=jnd_ret.columns[3].strip())


fig = plt.figure(figsize=[16,15])   ## create a figure 

fig.autofmt_xdate(rotation=90) 
fig.add_subplot(2,2,1)
jnd_ret[jnd_ret.columns[4]].plot(xlabel="Date", ylabel="Returns", title=jnd_ret.columns[4].strip())

fig.add_subplot(2,2,2) 
jnd_ret[jnd_ret.columns[5]].plot(xlabel="Date", ylabel="Returns", title=jnd_ret.columns[5].strip())

fig.add_subplot(2,2,3) 
jnd_ret[jnd_ret.columns[6]].plot(xlabel="Date", ylabel="Returns", title=jnd_ret.columns[6].strip())


#%% Calculate the cumulative returns using the 'cumprod()' function
cum_ret = (1 + jnd_ret).cumprod()       ### compute the cumulative returns 

#%% plot the figures 

fig = plt.figure(figsize=[16,15])   ## create a figure 

fig.autofmt_xdate(rotation=90) 
fig.add_subplot(2,2,1)
cum_ret[cum_ret.columns[0]].plot(xlabel="Date", ylabel="Cumulative Returns", title=cum_ret.columns[0].strip())

fig.add_subplot(2,2,2) 
cum_ret[cum_ret.columns[1]].plot(xlabel="Date", ylabel="Cumulative Returns", title=cum_ret.columns[1].strip())

fig.add_subplot(2,2,3) 
cum_ret[cum_ret.columns[2]].plot(xlabel="Date", ylabel="Cumulative Returns", title=cum_ret.columns[2].strip())

fig.add_subplot(2,2,4) 
cum_ret[cum_ret.columns[3]].plot(xlabel="Date", ylabel="Cumulative Returns", title=cum_ret.columns[3].strip())


fig = plt.figure(figsize=[16,15])   ## create a figure 

fig.autofmt_xdate(rotation=90) 
fig.add_subplot(2,2,1)
cum_ret[cum_ret.columns[4]].plot(xlabel="Date", ylabel="Cumulative Returns", title=cum_ret.columns[4].strip())

fig.add_subplot(2,2,2) 
cum_ret[cum_ret.columns[5]].plot(xlabel="Date", ylabel="Cumulative Returns", title=cum_ret.columns[5].strip())

fig.add_subplot(2,2,3) 
cum_ret[cum_ret.columns[6]].plot(xlabel="Date", ylabel="Cumulative Returns", title=cum_ret.columns[6].strip())

#%% create a boxplot for each portfolio


fig1, ax1 = plt.subplots()
ax1.boxplot(jnd_ret, notch=True, labels = [cum_ret.columns[0],
                                            cum_ret.columns[1],
                                            cum_ret.columns[2],
                                            cum_ret.columns[3],
                                            cum_ret.columns[4],
                                            cum_ret.columns[5],
                                            cum_ret.columns[6]
                                            ])

ax1.set_title('Box Plots Of All Portfolios')
ax1.legend([cum_ret.columns[0],
                                            cum_ret.columns[1],
                                            cum_ret.columns[2],
                                            cum_ret.columns[3],
                                            cum_ret.columns[4],
                                            cum_ret.columns[5],
                                            cum_ret.columns[6]
                                            ], fontsize='x-small')
plt.show()


#%% compute the annualized standard deviation for each portfolio 

std_ret_yr = jnd_ret.std().sort_values(ascending=False) * np.sqrt(252) 
print(f"Ranked Annual Standard Deviation Of Returns: \n\n{std_ret_yr}")


std_ret_dy = jnd_ret.std().sort_values(ascending=False)  
print(f"Ranked Daily Standard Deviation Of Returns: \n\n{std_ret_dy}")


#%%

# Risk Analysis
# Determine the risk of each portfolio:
# Create a box plot for each portfolio. 
# Calculate the standard deviation for all portfolios
# Determine which portfolios are riskier than the S&P 500
# Calculate the Annualized Standard Deviation

print(f"Standard deviation of the SP500 yearly --> {std_ret_yr['SP500']}")

more_risky = std_ret_yr[std_ret_yr>std_ret_yr['SP500']]

print(f"Portfolios riskier than the SP500 (yearly) are: \n\n{more_risky} \n")

print(f"Standard deviation of the SP500 daily --> {std_ret_dy['SP500']}")

more_risky_dy = std_ret_dy[std_ret_dy>std_ret_dy['SP500']]

print(f"Portfolios riskier than the SP500 (daily) are: \n\n{more_risky_dy} \n")


#%% compute the rolling std dev of all the portfolios

roll_std = jnd_ret.rolling(window=21).std()

#%% plot the figures 

fig = plt.figure(figsize=[16,15])   ## create a figure 

fig.autofmt_xdate(rotation=90) 
fig.add_subplot(2,2,1)
roll_std[roll_std.columns[0]].plot(xlabel="Date", ylabel="Roling Standard Deviation -- 21 Days", title=roll_std.columns[0].strip())

fig.add_subplot(2,2,2) 
roll_std[roll_std.columns[1]].plot(xlabel="Date", ylabel="Roling Standard Deviation -- 21 Days", title=roll_std.columns[1].strip())

fig.add_subplot(2,2,3) 
roll_std[roll_std.columns[2]].plot(xlabel="Date", ylabel="Roling Standard Deviation -- 21 Days", title=roll_std.columns[2].strip())

fig.add_subplot(2,2,4) 
roll_std[roll_std.columns[3]].plot(xlabel="Date", ylabel="Roling Standard Deviation -- 21 Days", title=roll_std.columns[3].strip())


fig = plt.figure(figsize=[16,15])   ## create a figure 

fig.autofmt_xdate(rotation=90) 
fig.add_subplot(2,2,1)
roll_std[roll_std.columns[4]].plot(xlabel="Date", ylabel="Roling Standard Deviation -- 21 Days", title=roll_std.columns[4].strip())

fig.add_subplot(2,2,2) 
roll_std[roll_std.columns[5]].plot(xlabel="Date", ylabel="Roling Standard Deviation -- 21 Days", title=roll_std.columns[5].strip())

fig.add_subplot(2,2,3) 
roll_std[roll_std.columns[6]].plot(xlabel="Date", ylabel="Roling Standard Deviation -- 21 Days", title=roll_std.columns[6].strip())

#%% compute the correlation across all portfolios

# Calculate the correlation
corr_ret = jnd_ret.corr()


#%% plot the correlation matrix

sns.heatmap(corr_ret, vmin=-1, vmax =1)

#%% compute the beta 
# Calculate variance of a single portfolio
var_sp500 = jnd_ret['SP500'].var() 

## covariance between all portfolios
cov_jnd = jnd_ret.cov()

sns.heatmap(cov_jnd, vmin=-0.00005, vmax = 0.00005, label="Correlation Matrix")

# Calculate covariance of a single portfolio
cov_soros = jnd_ret['SOROS FUND MANAGEMENT LLC'].cov(jnd_ret['SOROS FUND MANAGEMENT LLC'])

# Calculate variance of S&P 500
var_sp500

# Computing beta

## compute the covariance between a portfolio and the SP500 

cov1 = jnd_ret['SOROS FUND MANAGEMENT LLC'].cov(jnd_ret['SP500'])

beta1 = cov1 / var_sp500


# Plot beta trend

#%% # Use `ewm` to calculate the rolling window
## Try calculating the ewm with a 21-day half-life.

ewm_jnd = jnd_ret.ewm(halflife = 21).mean()

fig = plt.figure(figsize=[16,15])   ## create a figure 

fig.autofmt_xdate(rotation=90) 
fig.add_subplot(2,2,1)
ewm_jnd[ewm_jnd.columns[0]].plot(xlabel="Date", ylabel="Exponentially Weighted Mean 21 Day Half-Life", title=ewm_jnd.columns[0].strip())

fig.add_subplot(2,2,2) 
ewm_jnd[ewm_jnd.columns[1]].plot(xlabel="Date", ylabel="Exponentially Weighted Mean 21 Day Half-Life", title=ewm_jnd.columns[1].strip())

fig.add_subplot(2,2,3) 
ewm_jnd[ewm_jnd.columns[2]].plot(xlabel="Date", ylabel="Exponentially Weighted Mean 21 Day Half-Life", title=ewm_jnd.columns[2].strip())

fig.add_subplot(2,2,4) 
ewm_jnd[ewm_jnd.columns[3]].plot(xlabel="Date", ylabel="Exponentially Weighted Mean 21 Day Half-Life", title=ewm_jnd.columns[3].strip())


fig = plt.figure(figsize=[16,15])   ## create a figure 

fig.autofmt_xdate(rotation=90) 
fig.add_subplot(2,2,1)
ewm_jnd[ewm_jnd.columns[4]].plot(xlabel="Date", ylabel="Exponentially Weighted Mean 21 Day Half-Life", title=ewm_jnd.columns[4].strip())

fig.add_subplot(2,2,2) 
ewm_jnd[ewm_jnd.columns[5]].plot(xlabel="Date", ylabel="Exponentially Weighted Mean 21 Day Half-Life", title=ewm_jnd.columns[5].strip())

fig.add_subplot(2,2,3) 
ewm_jnd[ewm_jnd.columns[6]].plot(xlabel="Date", ylabel="Exponentially Weighted Mean 21 Day Half-Life", title=ewm_jnd.columns[6].strip())

#%% 


# Use the `mean` and `std` functions to calculate the annualized sharpe ratio
sharpe_ratios = (jnd_ret.mean() * 252) / (jnd_ret.std() * np.sqrt(252))
sharpe_ratios
shp = sharpe_ratios

#%% create a bar plot of the sharpe ratios
xlab = sharpe_ratios.index

fig1, br = plt.subplots(figsize=(22,18))

br.bar(height=shp, x = [shp.index[0], 
                        shp.index[1],
                        shp.index[2],
                        shp.index[3],
                        shp.index[4],
                        shp.index[5],
                        shp.index[6]
                        ])

#%% 
# Create Custom Portfolio
# In this section, you will build your own portfolio of stocks, calculate the returns, and compare the results to the Whale Portfolios and the S&P 500.

# Choose 3-5 custom stocks with at last 1 year's worth of historic prices and create a DataFrame of the closing prices and dates for each stock.
# Calculate the weighted returns for the portfolio assuming an equal number of shares for each stock
# Join your portfolio returns to the DataFrame that contains all of the portfolio returns
# Re-run the performance and risk analysis with your portfolio to see how it compares to the others
# Include correlation analysis to determine which stocks (if any) are correlated
# Choose 3-5 custom stocks with at last 1 year's worth of historic prices and create a DataFrame of the closing prices and dates for each stock.
# For this demo solution, we fetch data from three companies listes in the S&P 500 index.

# GOOG - Google, LLC

# AAPL - Apple Inc.

# COST - Costco Wholesale Corporation


#%%  create a separate portfolio 

#%% read in the data 

goog_pth = Path(basedir + "/Resources/Google_6yr.csv")
ndva_pth = Path(basedir + "/Resources/NVIDIA_6yr.csv")
ba_pth = Path(basedir + "/Resources/Boeing_6yr.csv")


go_df = pd.read_csv(goog_pth, index_col=("Date"))
nd_df = pd.read_csv(ndva_pth, index_col=("Date"))
ba_df = pd.read_csv(ba_pth, index_col=("Date"))

go_df = pd.DataFrame(go_df['Close'].rename("Google"))
nd_df = pd.DataFrame(nd_df['Close'].rename("NVIDIA"))
ba_df = pd.DataFrame(ba_df['Close'].rename("Boeing"))

#%% clean thedata 

## Count nulls 
go_nulls = go_df.isnull().sum()
nd_nulls = nd_df.isnull().sum()
ba_nulls = ba_df.isnull().sum()

print(f"Nulls: Google={go_nulls}, NVIDIA={nd_nulls}, Cost={cost_nulls}, Boeing={ba_nulls}")

#%% drop the nulls 

go = go_df.pct_change().dropna() 
nd = nd_df.pct_change().dropna() 
ba = ba_df.pct_change().dropna() 

#%% combine the data 
prt_ret = pd.concat([go, nd, ba], axis="columns", join = "inner")

tt = pd.to_datetime(prt_ret.index)
prt_ret.index = tt
prt_ret = prt_ret.sort_index(ascending=False)

#%% plot the data 
go_df.plot()
nd_df.plot()
ba_df.plot()

#%% plot the returns 
go.plot()
nd.plot() 
ba.plot() 

#%%# Set weights
weights = [1/3, 1/3, 1/3]

# Calculate portfolio return
prt_ret_w = prt_ret.dot(weights)

prt_ret_w = pd.DataFrame(prt_ret_w.rename("new_port"))

# Display sample data
prt_ret_w.plot()

#%% Join your returns DataFrame to the original returns DataFrame

## normalize the date time index to be aligned with the other portfolio dataframe

prt_ret.index = prt_ret.index.normalize()
prt_ret_w = prt_ret.dot(weights)
all_ret = pd.concat([jnd_ret, prt_ret_w], axis="columns", join = "inner")


#%%  sort the std -- annualized 

all_ret_std = all_ret.std().sort_values()*np.sqrt(252)


#%%# Calculate rolling standard deviation
all_ret_roll_std = all_ret.rolling(window=21).std()

# Plot rolling standard deviation
all_ret_roll_std.plot()

fig = plt.figure(figsize=[16,15])   ## create a figure 

fig.autofmt_xdate(rotation=90) 
fig.add_subplot(2,2,1)
all_ret_roll_std[all_ret_roll_std.columns[0]].plot(xlabel="Date", ylabel="Roling Standard Deviation -- 21 Days", title=all_ret_roll_std.columns[0].strip())

fig.add_subplot(2,2,2) 
all_ret_roll_std[all_ret_roll_std.columns[1]].plot(xlabel="Date", ylabel="Roling Standard Deviation -- 21 Days", title=all_ret_roll_std.columns[1].strip())

fig.add_subplot(2,2,3) 
all_ret_roll_std[all_ret_roll_std.columns[2]].plot(xlabel="Date", ylabel="Roling Standard Deviation -- 21 Days", title=all_ret_roll_std.columns[2].strip())

fig.add_subplot(2,2,4) 
all_ret_roll_std[all_ret_roll_std.columns[3]].plot(xlabel="Date", ylabel="Roling Standard Deviation -- 21 Days", title=all_ret_roll_std.columns[3].strip())


fig = plt.figure(figsize=[16,15])   ## create a figure 

fig.autofmt_xdate(rotation=90) 
fig.add_subplot(2,2,1)
all_ret_roll_std[all_ret_roll_std.columns[4]].plot(xlabel="Date", ylabel="Roling Standard Deviation -- 21 Days", title=all_ret_roll_std.columns[4].strip())

fig.add_subplot(2,2,2) 
all_ret_roll_std[all_ret_roll_std.columns[5]].plot(xlabel="Date", ylabel="Roling Standard Deviation -- 21 Days", title=all_ret_roll_std.columns[5].strip())

fig.add_subplot(2,2,3) 
all_ret_roll_std[all_ret_roll_std.columns[6]].plot(xlabel="Date", ylabel="Roling Standard Deviation -- 21 Days", title=all_ret_roll_std.columns[6].strip())

fig.add_subplot(2,2,4) 
all_ret_roll_std[all_ret_roll_std.columns[7]].plot(xlabel="Date", ylabel="Roling Standard Deviation -- 21 Days", title="Weighted Portfolio")


#%% calculate and plot the correlation 


# Calculate the correlation
corr_all_ret = all_ret.corr()


#%% plot the correlation matrix

sns.heatmap(corr_all_ret, vmin=-1, vmax =1)

#%% Calculate and plot the beta

# Calculate and Plot Rolling 60-day Beta for Your Portfolio compared to the S&P 500

wgt = all_ret.iloc[:,7]


wgt_roll_cov = all_ret.iloc[:,7].rolling(window=60).cov(all_ret['SP500']).dropna()
sp_roll_var = all_ret['SP500'].rolling(window=60).var().dropna()

beta_wgt = wgt_roll_cov / sp_roll_var

beta_wgt.plot()

#%% Using the daily returns, calculate and visualize the Sharpe ratios using a bar plot
# Use the `mean` and `std` functions to calculate the annualized sharpe ratio
sharpe_ratios = (all_ret.mean() * 252) / (all_ret.std() * np.sqrt(252))

fig1, br = plt.subplots(figsize=(22,18))

br.bar(height=sharpe_ratios, x = [sharpe_ratios.index[0], 
                        sharpe_ratios.index[1],
                        sharpe_ratios.index[2],
                        sharpe_ratios.index[3],
                        sharpe_ratios.index[4],
                        sharpe_ratios.index[5],
                        sharpe_ratios.index[6],
                        "Weighted Portfolio",
                        ])
#%%




