#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import yfinance as yf
import pandas as pd


# In[2]:


#Data cleaning 
file_names_lst = ["DIA.csv", "FEZ.csv", "IJH.csv", "IWD.csv", "IWF.csv",
                  "IWM.csv", "IWN.csv", "IWO.csv", "MDY.csv", "OEF.csv", "QQQ.csv", "SPY.csv", "VTI.csv"]
equity_names_lst = ["DIA", "FEZ", "IJH", "IWD", "IWF",
                    "IWM", "IWN", "IWO", "MDY", "OEF", "QQQ", "SPY", "VTI"]
uni_daily_prices = pd.DataFrame()
for i in range(len(file_names_lst)):
    file_name = file_names_lst[i]
    equity_name = equity_names_lst[i]
    df = pd.read_csv(file_name)
    df = df[['Date', 'Adj Close']]
    df['Date'] = pd.to_datetime(df['Date'])
    cutoff_date = pd.to_datetime('2005-01-01')
    df = df.loc[df['Date'] > cutoff_date, :]
    if i == 0:
        uni_daily_prices['Date'] = df['Date']
        uni_daily_prices.set_index('Date', inplace=True)
    df.set_index('Date', inplace=True)
    uni_daily_prices[equity_name] = df["Adj Close"]
uni_daily_returns_df = uni_daily_prices.pct_change()
uni_monthly_prices = uni_daily_prices.resample('M').last()
uni_monthly_returns_df = uni_monthly_prices.pct_change()
uni_daily_returns_df = uni_daily_returns_df.iloc[1:]
# uni_daily_returns_df.to_csv('index_return.csv')


# In[3]:


# Find the minimum value of SSD(sum of squared difference) among each pair of return of indexes
def SSD(df):
    ssd_df = pd.DataFrame(index = ['SSD'])
    for i in range(len(df.columns)):
        for j in range(i+1, len(df.columns)):
            # calculate SSD between columns i and j
            col1 = df.iloc[:, i]
            col2 = df.iloc[:, j]
            ssd = ((col1 - col2) ** 2).sum()
            
            # store result in ssd_df
            colname = f"{df.columns[i]}-{df.columns[j]}"
            ssd_df[colname] = [ssd]
    
    return ssd_df
ssd_daily_df = SSD(uni_daily_returns_df).T
min_ssd_daily = ssd_daily_df.idxmin(axis=0)[0]
ssd_daily_df.sort_values(by='SSD', ascending=True, inplace=True)

print(ssd_daily_df)

print(f"Minimum SSD value(daily) is {ssd_daily_df['SSD'][min_ssd_daily]} in column '{min_ssd_daily}'")
# Minimum SSD value(daily) is 0.009802063661051762 in column 'SPY-VTI' when the indexes are ['DIA','FEZ','IJH','IWD','IWF','IWM','IWN','IWO','MDY','OEF','QQQ','SPY','VTI']


# In[4]:


# Use Kendall Tau to find the pair.
from scipy.stats import kendalltau

def Kendall_Tau(df):
    kendall_df = pd.DataFrame(index = ['KT'])
    # iterate over column pairs
    for i in range(len(df.columns)):
        for j in range(i+1, len(df.columns)):
            # calculate Kendall tau correlation between columns i and j
            col1 = df.iloc[:, i]
            col2 = df.iloc[:, j]
            tau, pval = kendalltau(col1, col2)
            # store result in kendall_df
            colname = f"{df.columns[i]}-{df.columns[j]}"
            kendall_df[colname] = [tau]
    # print results
    return kendall_df

kt_daily_df = Kendall_Tau(uni_daily_returns_df).T
max_kt_daily = kt_daily_df.idxmax(axis=0)[0]
kt_daily_df.sort_values(by='KT', ascending=False, inplace=True)

print(kt_daily_df)
print(f"Maximum KT value(daily) is {kt_daily_df['KT'][max_kt_daily]} in column '{max_kt_daily}'")
# Maximum KT value(daily) is 0.9443723187207665 in column 'IJH-MDY'


# In[5]:


#approach 1: simple OLS method to find hedge ratio
# the SPY-VTI pair
SPY = uni_daily_returns_df['SPY']
VTI = uni_daily_returns_df['VTI']
cov_1 = np.cov(SPY, VTI)[0][1]
var_SPY = np.var(SPY)
var_VTI = np.var(VTI)
hedge_linear_1 = cov_1/var_SPY

#the IJH-MDY pair
hedge_linear_2= np.cov(uni_daily_returns_df['IJH'],uni_daily_returns_df['MDY'])[0][1]/np.var(uni_daily_returns_df['IJH'])


# In[6]:



import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

dist1 = norm(loc=SPY.mean(), scale=SPY.std())
dist2 = norm(loc=VTI.mean(), scale=VTI.std())

# Define the sample size
n = 200

# Generate random samples from the two distributions
sample1 = dist1.rvs(size=n)
sample2 = dist2.rvs(size=n)

# Calculate the CDF
x = np.linspace(-0.1, 0.1, num=1000)
cdf1 = dist1.cdf(x)
cdf2 = dist2.cdf(x)

# Map the samples to their quantile domain
quant1 = np.searchsorted(np.sort(sample1), sample1)/n
quant2 = np.searchsorted(np.sort(sample2), sample2)/n

# Plot the quantile-quantile data
plt.scatter(quant1, quant2)
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.xlabel('Quantiles of Sample SPY')
plt.ylabel('Quantiles of Sample VTI')
plt.show()


# In[7]:


#appraoch 2:lookback window for the rolling regression
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
pair_1 = pd.merge(SPY,VTI,on = 'Date')

roll_reg = RollingOLS.from_formula('VTI ~ SPY -1', window=25, data=pair_1)
model = roll_reg.fit()


# In[8]:


print(model.params[25:])
fig = model.plot_recursive_coefficient(variables=['SPY'])
plt.xlabel('Time step')
plt.ylabel('Coefficient value')
plt.show()


# In[9]:


#approach 2: Recursive ordinary least squares (aka expanding window rolling regression) 
reg_rls = sm.RecursiveLS.from_formula(
    'VTI ~ SPY-1 ', pair_1)
model_rls = reg_rls.fit()
print(model_rls.summary())

fig = model_rls.plot_recursive_coefficient(range(reg_rls.k_exog), legend_loc='upper right')
ax_list = fig.axes
for ax in ax_list:
    ax.set_xlim(0, None)
ax_list[-1].set_xlabel('Time step')
ax_list[0].set_title('Coefficient value')


# In[11]:


#appraoch 3: Kalman Filter
from __future__ import print_function
from pykalman import KalmanFilter
def draw_date_coloured_scatterplot(pair_1):
    """
    Create a scatterplot of the two indicies prices, which is
    coloured by the date of the price to indicate the 
    changing relationship between the sets of prices    
    """
    # Create a yellow-to-red colourmap where yellow indicates early dates and red indicates later dates
    plen = len(pair_1)
    colour_map = plt.cm.get_cmap('YlOrRd')    
    colours = np.linspace(0.1, 1, plen)
    
    # Create the scatterplot object
    scatterplot = plt.scatter(
        pair_1['SPY'], pair_1['VTI'], 
        s=30, c=colours, cmap=colour_map, 
        edgecolor='k', alpha=0.8
    )
    
    # Add a colour bar for the date colouring and set the corresponding axis tick labels to equal string-formatted dates
    colourbar = plt.colorbar(scatterplot)
    colourbar.ax.set_yticklabels(
        [str(p.date()) for p in pair_1[::plen//9].index]
    )
    plt.xlabel(pair_1.columns[0])
    plt.ylabel(pair_1.columns[1])
    plt.show()
draw_date_coloured_scatterplot(pair_1)


# In[13]:


def calc_slope_intercept_kalman(pair_1):
    """
    Utilise the Kalman Filter from the pyKalman package
    to calculate the slope and intercept of the regressed
    ETF prices.
    """
    delta = 1e-5
    trans_cov = delta / (1 - delta) * np.eye(2)
    obs_mat = np.vstack(
        [pair_1['SPY'], np.ones(pair_1['SPY'].shape)]
    ).T[:, np.newaxis]
    
    kf = KalmanFilter(
        n_dim_obs=1, 
        n_dim_state=2,
        initial_state_mean=np.zeros(2),
        initial_state_covariance=np.ones((2, 2)),
        transition_matrices=np.eye(2),
        observation_matrices=obs_mat,
        observation_covariance=1.0,
        transition_covariance=trans_cov
    )
    
    state_means, state_covs = kf.filter(pair_1['VTI'].values)
    return state_means, state_covs    
    
state_means, state_covs = calc_slope_intercept_kalman(pair_1)


# In[15]:


def draw_slope_intercept_changes(pair_1, state_means):
    """
    Plot the slope and intercept changes from the 
    Kalman Filte calculated values.
    """
    pd.DataFrame(
        dict(
            slope=state_means[:, 0], 
            intercept=state_means[:, 1]
        ), index=pair_1.index
    ).plot(subplots=True)
    plt.show()

draw_slope_intercept_changes(pair_1, state_means)

