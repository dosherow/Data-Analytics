# Drew Osherow Assignment 3
# ITP449, Spring 2019
# 6130663389
# dosherow@usc.edu

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

os.chdir('/Users/drewosherow/Desktop/spring2019/itp449/python')

# read dataset into python using pandas

data = pd.read_csv('avocado.csv')

# store the data in a dataFrame with named indices

frame = pd.DataFrame(data.values,
                     columns=['Index', 'Date', 'AveragePrice', 'Total Volume', '4046', '4225', '4770', 'Total Bags',
                              'Small Bags', 'Large Bags', 'XLarge Bags', 'type', 'year', 'region'],
                     index=[np.arange(1,18250)])
pd.set_option('display.max_columns', None)

print(frame.head())

# plot price of avocados over time

data['TotalPrice'] = data['AveragePrice']
total_price = data.groupby('Date')['TotalPrice'].mean()

frame.plot(y=total_price, x='Date', linewidth = 2, fontsize = 15)
plt.xlabel('Time', fontsize = 15)
plt.ylabel('Price', fontsize = 15)
plt.show()

# plot total volume of avocados sold along with price over time

# perform smoothing on both plots

# what are overall trends in price over entire range of time (2015-2018)?

# what are overall trends in total volume over entire range of time?

# what are yearly trends in price?

# what are yearly trends in total volume?

# what is relationship between price and total volume?

