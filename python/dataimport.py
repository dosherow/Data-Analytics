import os
import pandas
import matplotlib.pyplot as plt
# change directory to where all ITP449 files are stored
os.chdir('/Users/drewosherow/Desktop/spring2019/itp449/python')

# read and store a csv file
puzzleCompletionTime = pandas.read_csv('grade5ScienceFairProject.csv')
print(puzzleCompletionTime)