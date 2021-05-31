# -*- coding: UTF-8 -*-
"""PyRamen Homework Starter."""

#%% @TODO: Import libraries
import csv
import copy
##import pandas as pd 
from pathlib import Path

#%% @TODO: Set file paths for menu_data.csv and sales_data.csv
menu_filepath = Path('/home/oodaye/Fintech/HW_2_FinTech/PyRamen/Resources/menu_data.csv')
sales_filepath = Path('/home/oodaye/Fintech/HW_2_FinTech/PyRamen/Resources/sales_data.csv')

#%% @TODO: Initialize list objects to hold our menu and sales data
menu = []
menu_hdr = []
sales = []

#%%

with open(menu_filepath, "r") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    
    menu_hdr = next(csv_reader)            # save the header 
    for lines in csv_reader:
        menu.append(lines)


#%% @TODO: Read in the sales data into the sales list

with open(sales_filepath, "r") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    
    sales_hdr = next(csv_reader)    ## save the header 
    for lines in csv_reader:
        sales.append(lines)


#%% # @TODO: Initialize dict object to hold our key-value pairs of items and metrics
report = {}


# Initialize a row counter variable
row_count = 0

init_metrics = {
"01-count": 0,
"02-revenue": 0,
"03-cogs": 0,
"04-profit": 0,
}

#%%
# @TODO: Loop over every row in the sales list object

for sale in sales: 
    sales_item  = sale[4]
    if sales_item not in report: 
        report[sales_item] = copy.deepcopy(init_metrics)    
   
    
#%%    
    
for ii in range(len(sales)):
    # Line_Item_ID,Date,Credit_Card_Number,Quantity,Menu_Item
    # @TODO: Initialize sales data variables

    sale = sales[ii]
    quantity = float(sale[3])
    sales_item  = sale[4]
       
    for menu_item in menu:   ## loop through the menu 
        item = menu_item[0]
        price = float(menu_item[3])
        cost = float(menu_item[4])
        profit = price - cost   # compute the profit

        if item == sales_item:
            print(f"MATCH FOUND -- {sales_item} does equal {item}!")
            report[sales_item]["01-count"] += quantity
            report[sales_item]["02-revenue"] += price * quantity
            report[sales_item]["03-cogs"] += cost * quantity
            report[sales_item]["04-profit"] += profit * quantity

        else:
            print(f"NO MATCH FOUND! -- {sales_item} does not equal {item}!")

    row_count += 1  ## increment row count 
    
#### end of sales: loop

#%% write out the file with the report 

with open('PyRamen_Output_4.txt', 'wt') as ff:
    
    print("RAMEN SALES REPORT", file = ff)
    print("---------------------------------------------", file = ff)
        
    for rep in report:
        print(f"{rep} -- {report[rep]}",file = ff)

