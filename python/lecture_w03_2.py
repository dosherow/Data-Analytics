import os
import pandas
import matplotlib.pyplot as plt
import seaborn as sb

# change the directory to where all ITP449 files are stored
os.chdir('/Users/drewosherow/Desktop/spring2019/itp449/python')

# read and store a csv  file
puzzleCompletionTime = pandas.read_csv('grade5ScienceFairProject.csv')

sb.set_style("whitegrid")
ax = sb.swarmplot(
    x = "Music",
    y = "Seconds",
    data=puzzleCompletionTime,
    size=10)

ax = sb.boxplot(
    x = "Music",
    y = "Seconds",
    data = puzzleCompletionTime
)

plt.show()