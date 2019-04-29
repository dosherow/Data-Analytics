# Drew Osherow Assignment 1
# ITP449, Spring 2019
# 6130663389
# dosherow@usc.edu

import os
import pandas
import matplotlib.pyplot as plt
import seaborn as sb

# change the directory to where all ITP449 files are stored
os.chdir('/Users/drewosherow/Desktop/spring2019/itp449/python')

# read the dataset into Python using Pandas
whiteHouseSalary = pandas.read_csv('2010_Report_to_Congress_on_White_House_Staff.csv')

# compute and display the following metrics for all white house staffer salaries:
# mean, standard deviation, median, minimum, maximum

mean = whiteHouseSalary['Salary'].mean()
std = whiteHouseSalary['Salary'].std()
median = whiteHouseSalary['Salary'].median()
min = whiteHouseSalary['Salary'].min()
max = whiteHouseSalary['Salary'].max()


print('Mean salary: ' + str(mean))
print('Standard deviation of salary: ' + str(std))
print('Median salary: ' + str(median))
print('Minimum salary: ' + str(min))
print('Maximum salary: ' + str(max))

# compute and display male and female white house staffer salaries: mean, std, median, min, max

group_mean = whiteHouseSalary.groupby(['Gender']).mean()
group_std = whiteHouseSalary.groupby(['Gender']).std()
group_median = whiteHouseSalary.groupby(['Gender']).median()
group_min = whiteHouseSalary.groupby(['Gender']).min()['Salary']
group_max = whiteHouseSalary.groupby(['Gender']).max()['Salary']

print('Mean salaries, by Gender: ' + str(group_mean))
print('Standard deviation of salaries, by Gender: ' + str(group_std))
print('Median salaries, by Gender: ' + str(group_median))
print('Minimum salary, by Gender: ' + str(group_min))
print('Maximum salary, by Gender: ' + str(group_max))

# Display histogram of White House staffer salaries


plt.hist(whiteHouseSalary, bins=5, rwidth=0.8, color='#000000')

plt.show()

# Display the swarmplot of White House staffer salaries

sb.set_style("whitegrid")
ax = sb.swarmplot(
    x = "Gender",
    y = "Salary",
    data=whiteHouseSalary,
    size=5
)

# Display boxplot superimposed of White House staffer salaries

ax = sb.boxplot(
    x = "Gender",
    y = "Salary",
    data=whiteHouseSalary
)

plt.show()

# Key Insights:
# According to the dataset, it seems as though Male White House staffers make more money on average than Female White House staffers,
# although the results may be skewed due to a few Female employees having $0 for Salary in the CSV. This may be for
# unknown reasons, but we may infer that they no longer are employed by the White House. Other insights
# we can gather are that the IQR covers a wider range of salaries for Males compared to Females
# responsible for the greater spread between individual salaries. Male salaries also dominate the two genders
# Q3 regions in terms of density, spread, and higher mean.