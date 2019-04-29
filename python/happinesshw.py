# Drew Osherow
# Happiness HW
# dosherow@usc.edu
# 6130663389


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from scipy import stats
import statsmodels.api as sm
# change the directory to where all ITP449 files are stored
os.chdir('/Users/drewosherow/Desktop/spring2019/itp449/python')

# download the attached CSV file
# read datasets into python using pandas
# store data in a dataframe with named indices
happiness_df = pd.read_csv('2015.csv')
df = DataFrame(happiness_df)
df.columns=['Country','Region','Happiness_Rank','Happiness_Score',
            'Economy','Family','Health','Freedom','Trust','Generosity','Dystopia']

print(df.head)
print(df.dtypes)


# checking for linearity between variables and dependent variable happiness score
# economy vs. happiness score
plt.scatter(df['Economy'], df['Happiness_Score'], color='red')
plt.title('Economy vs. Happiness Score', fontsize=12)
plt.xlabel('Economy', fontsize=12)
plt.ylabel('Happiness Score', fontsize=12)
plt.grid(True)
plt.show()

# family vs. happiness score
plt.scatter(df['Family'], df['Happiness_Score'], color='red')
plt.title('Family vs. HappinessScore', fontsize=12)
plt.xlabel('Family', fontsize=12)
plt.ylabel('Happiness Score', fontsize=12)
plt.grid(True)
plt.show()

# Health vs. happiness score
plt.scatter(df['Health'], df['Happiness_Score'], color='red')
plt.title('Health vs. HappinessScore', fontsize=12)
plt.xlabel('Health', fontsize=12)
plt.ylabel('Happiness Score', fontsize=12)
plt.grid(True)
plt.show()

# freedom vs. happiness score
plt.scatter(df['Freedom'], df['Happiness_Score'], color='red')
plt.title('Freedom vs. Happiness Score', fontsize=12)
plt.xlabel('Freedom', fontsize=12)
plt.ylabel('Happiness Score', fontsize=12)
plt.grid(True)
plt.show()

# Trust vs. happiness score
plt.scatter(df['Trust'], df['Happiness_Score'], color='red')
plt.title('Trust vs. Happiness Score', fontsize=12)
plt.xlabel('Trust', fontsize=12)
plt.ylabel('Happiness Score', fontsize=12)
plt.grid(True)
plt.show()

# generosity vs. happiness score
plt.scatter(df['Generosity'], df['Happiness_Score'], color='red')
plt.title('Generosity vs. Happiness Score', fontsize=12)
plt.xlabel('Generosity', fontsize=12)
plt.ylabel('Happiness Score', fontsize=12)
plt.grid(True)
plt.show()

# dystopia vs. happiness score
plt.scatter(df['Dystopia'], df['Happiness_Score'], color='red')
plt.title('Dystopia vs. Happiness Score', fontsize=12)
plt.xlabel('Dystopia', fontsize=12)
plt.ylabel('Happiness Score', fontsize=12)
plt.grid(True)
plt.show()

# what factors affect happiness the most?
# describe the relationship between each factor and the happiness score

X1 = df['Economy']
y = df['Happiness_Score']

# regression summary using statsmodels to see if any p-values are too large to be used
# and to look at r-squared values of each independent variable
# economy vs. happiness score
X2 = sm.add_constant(X1)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())

# r-squared = .617
# intercept = 3.3769
# coefficient = 2.1489


# family vs. happiness score
X3 = df['Family']

X4 = sm.add_constant(X3)
est3 = sm.OLS(y, X4)
est4 = est3.fit()
print(est4.summary())

# r-squared = .405
# intercept = 3.1218
# coefficient = 2.2708

# health vs. happiness score
X5 = df['Health']

X6 = sm.add_constant(X5)
est5 = sm.OLS(y, X6)
est6 = est5.fit()
print(est6.summary())

# r-squared = .560
# intercept = 3.3168
# coefficient = 3.5415

# freedom vs. happiness score
X7 = df['Freedom']

X8 = sm.add_constant(X7)
est7 = sm.OLS(y, X8)
est8 = est7.fit()
print(est8.summary())

# r-squared = .314
# intercept = 3.6638
# coefficient = 4.2374

# trust vs. happiness score
X9 = df['Trust']

X10 = sm.add_constant(X9)
est9 = sm.OLS(y, X10)
est10 = est9.fit()
print(est10.summary())

# r-squared = .653
# intercept = 4.8113
# coefficient = 4.1505

# generosity vs. happiness score
X11 = df['Generosity']

X12 = sm.add_constant(X11)
est11 = sm.OLS(y, X12)
est12 = est11.fit()
print(est12.summary())

# r-squared = .027
# intercept = 5.0283
# coefficient = 1.4138

# dystopia vs. happiness score
X13 = df['Dystopia']

X14 = sm.add_constant(X13)
est13 = sm.OLS(y, X14)
est14 = est13.fit()
print(est14.summary())

# r-squared = .240
# intercept = 3.3110
# coefficient = 0.9842


# running corelation between each independent variable in respect to happiness score
# as we see, economy, health, and family are biggest factors.
corr_matrix = df.corr()
print(corr_matrix['Happiness_Score'].sort_values(ascending=False))

# after running regressions for each factor and looking at all of the information from
# the summaries, it looks like economy, family, and health are the most important
# factors for happiness score. they have relatively high r^2 values, and even though
# the trust r^2 value is very high, looking at the scatterplot for that variable
# there isn't much linearity we see.

# are the factors affecting happiness different across different regions?
regionsdf = df.groupby('Region')['Happiness_Score']
print(regionsdf.describe())

# X15 = df['Region']
#
# X16 = sm.add_constant(X15)
# est15 = sm.OLS(y, X16)
# est16 = est15.fit()
# print(est16.summary())



# develop and optimize a model to predict happiness score

X = df[['Economy','Family','Health','Freedom','Trust','Generosity','Dystopia']]
y = df['Happiness_Score']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
from sklearn.linear_model import LinearRegression
lg = LinearRegression()
lg.fit(X_train,y_train)

predictions = lg.predict(X_test)
plt.scatter(y_test,predictions)
plt.show()

