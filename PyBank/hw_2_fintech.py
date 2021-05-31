## HW 2 fintech

#%% imports 
import pandas as pd
import os
import datetime as dt
## import scipy as sc
import numpy as np 

#%% set current working directory 
os.chdir("/home/oodaye/Fintech/HW_2_FinTech")

#%% home work instructions 

# ##Your task is to create a Python script that analyzes the records to calculate each of the following:

# The total number of months included in the dataset.
# The net total amount of Profit/Losses over the entire period.
# The average of the changes in Profit/Losses over the entire period.
# The greatest increase in profits (date and amount) over the entire period.
# The greatest decrease in losses (date and amount) over the entire period.
# Your resulting analysis should look similar to the following:
# Financial Analysis
# ----------------------------
# Total Months: 86
# Total: $38382578
# Average  Change: $-2315.12
# Greatest Increase in Profits: Feb-2012 ($1926159)
# Greatest Decrease in Profits: Sep-2013 ($-2196167)



#%%  read in the file 
fname = "PyBank.csv"
df = pd.read_csv(fname) 

## convert the first column to date time object 
df['Date'] = df['Date'].apply(lambda x: dt.datetime.strptime(x,'%b-%Y'))
df['year'] = df['Date'].map(lambda x: x.strftime('%Y'))
df['month'] = df['Date'].map(lambda x: x.strftime('%b'))

## get the years listed in the file 
yrs = pd.unique(df['Date'].map(lambda x: x.strftime('%Y')))


#%% compute the number of months 

## compute the number of months 
num_months = df.groupby('year')['month'].count().sum()
    
#%% compute the total 


tot = df['Profit/Losses'].sum() 

#%% compute the average change 

delta = np.ediff1d(df['Profit/Losses'])
avg_change = delta.mean()
avg_change = np.round(avg_change,2)   ## round to 2 decimal places 
    
#%% find the greatest increase in profits 
max_prof_inc_indx = pd.DataFrame(np.ediff1d(df['Profit/Losses'])).idxmax()

max_prof_inc = delta[max_prof_inc_indx]

mx_year = df['year'][max_prof_inc_indx+1]
mx_month = df['month'][max_prof_inc_indx+1]

mx_year_out = mx_year.values[0]
mx_month_out= mx_month.values[0]

#%% find the greatest decrease in profits 

max_prof_dec_indx = pd.DataFrame(np.ediff1d(df['Profit/Losses'])).idxmin()

max_prof_dec = delta[max_prof_dec_indx]

mn_year = df['year'][max_prof_dec_indx+1]
mn_month = df['month'][max_prof_dec_indx+1]

mn_year_out = mn_year.values[0]
mn_month_out= mn_month.values[0]


#%% write the results to a file  

print("Financial Analysis")
print("-------------------------")
print(f"Total Months: {num_months}")
print(f"Total: ${tot}")
print(f"Average  Change: ${avg_change}")
print(f"Greatest Increase in Profits: {mx_month_out}-{mx_year_out} (${max_prof_inc[0]})")
print(f"Greatest Decrease in Profits: {mn_month_out}-{mn_year_out} (${max_prof_dec[0]})")


with open('PyBank_Output_4.txt', 'wt') as ff:
    print("Financial Analysis", file = ff)
    print("-------------------------", file = ff)
    print(f"Total Months: {num_months}", file = ff)
    print(f"Total: ${tot}", file = ff)
    print(f"Average  Change: ${avg_change}", file = ff)
    print(f"Greatest Increase in Profits: {mx_month_out}-{mx_year_out} (${max_prof_inc[0]})", file = ff)
    print(f"Greatest Decrease in Profits: {mn_month_out}-{mn_year_out} (${max_prof_dec[0]})", file = ff)


   


