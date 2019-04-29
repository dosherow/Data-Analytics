# Drew Osherow Assignment 2
# ITP449, Spring 2019
# 6130663389
# dosherow@usc.edu

import os
import pandas as pd

# change the directory to where all ITP449 files are stored
os.chdir('/Users/drewosherow/Desktop/spring2019/itp449/python')

# read the dataset into Python using Pandas, store data in a dataframe
titanic = pd.read_csv('TitanicTrain.csv')

# Compute and display the following metrics for all Titanic passengers: mean and standard
# deviation of age, percentage of passenger who survived,
# percent of passengers who perished, percent of passengers in each class.
mean = titanic['Age'].mean()
std = titanic['Age'].std()
survived = titanic['Survived'].mean()
perished = 1 - survived
row = titanic.shape[0]
class1 = titanic['Pclass'].value_counts(normalize=True) * 100
print('Mean Age of Titanic Passengers:' + ' ' + str(mean))
print('Standard Deviation of Ages of Titanic Passengers:' + ' ' + str(std))
print('Percentage of Passengers Who Survived:' + ' ' + str(survived * 100))
print('Percentage of Passengers Who Perished:' + ' ' + str(perished * 100))
print('Percentage of Passengers In Each Class: \n' + str(class1))

# Create a new column and populate it with 'child' (for passengers under 18),
# 'adult' (for passengers 18 and older) or 'unknown' (for passengers with a missing age field)
def is_child(age):
    if age < 18:
        return str('child')
    elif age >= 18:
        return str('adult')
    else:
        return str('unknown')
titanic['Child/Adult'] = titanic['Age'].apply(is_child)

# Compute and display the following metrics for passengers who survived and
# perished: percent male and female, percent adult and child (under 18), percent
# from each class, percent traveling with family.
survived_gender = titanic.groupby('Sex')['Survived'].mean()
survived_age = titanic.groupby('Child/Adult')['Survived'].mean()
perished_gender = 1 - survived_gender
perished_age = 1 - survived_age
survived_class = titanic.groupby('Pclass')['Survived'].mean()
perished_class = 1 - survived_class
titanic['Family'] = (titanic['SibSp'] > 0) | (titanic['Parch'] > 0)
survived_family = titanic.groupby('Family')['Survived'].mean()
perished_family = 1 - survived_family
print('Percentage of Passengers Who Survived By Gender: \n' + str(survived_gender))
print('Percentage of Passengers Who Perished By Gender: \n' + str(perished_gender))
print('Percentage of Passengers Who Survived By Age Category: \n' + str(survived_age))
print('Percentage of Passengers Who Perished By Age: \n' + str(perished_age))
print('Percentage of Passengers Who Survived By Class: \n' + str(survived_class))
print('Percentage of Passengers Who Perished By Class: \n' + str(perished_class))
print('Percentage of Passengers Who Survived With/Without Family: \n' + str(survived_family))
print('Percentage of Passengers Who Perished With/Without Family: \n' + str(perished_family))

