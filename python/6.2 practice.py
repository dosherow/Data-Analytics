import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# read in files and display info and first 5 rows for all columns
# os.chdir('/Users/arpi-admin/Documents/ITP449_Files')
#
# df = pd.read_csv('VDayTrends.csv')
# print(df.info())
# print(df.head())



# convert month column to a datetime data type
# set month column as dataframe index

# df['Month'] = pd.to_datetime(df['Month'])
# df.set_index('Month', inplace=True)
# print(df.info())
# print(df.head())





# plot the data as two line plots on the same figure.
# x axis represents the year while y axis represents pop of each term

# df.plot(figsize=(18,8), linewidth=4, fontsize=15)
# plt.xlabel('Year', fontsize=15)
# plt.ylabel('Popularity', fontsize=15)
# plt.show()








# plot data for only search term lonely.
# xaxis represents the year while the y-axis represents popularity of each term
# df[['Lonely']].plot(figsize=(18,8), linewidth=4, fontsize=15)
# plt.xlabel('Year', fontsize=15)
# plt.ylabel('Popularity', fontsize=15)
# plt.show()







# plot data for only search term lonely and apply
# a smoothing function to it.

lonely = df[['Lonely']]

lonely.rolling(12).mean().plot(figsize=(18,8), linewidth=4, fontsize=15)
plt.xlabel('Year', fontsize=15)
plt.ylabel('Popularity', fontsize=15)
plt.show()