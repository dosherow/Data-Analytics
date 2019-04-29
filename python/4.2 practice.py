# create rock, paper, scissors game with following output:
# rock (r), paper (p), scissors (s)?
# Try again!
# you picked paper and the computer picked rock

import numpy as np
import pandas as pd

# things = ['Rock', 'Paper', 'Scissors']
#
# player = input("Rock (r), paper (p), scissors (s)? ")
#
#
#
# if player == 'r':
#     playerIndex = 0
# elif player == 'p':
#     playerIndex = 1
# elif player == 's':
#     playerIndex = 2
# else:
#     print("Try again!")
#     playerIndex = 3
#
# computerIndex = np.random.randint(low=0, high=2)
#
# if playerIndex != 3:
#     print("You picked", things[playerIndex], "and the computer picked", things[computerIndex])






# series1 = pd.Series([4, 7, -5, 3])
#
# print("Series: \n", series1)
# print("Series values: \n", series1.values)
# print("Series index: \n", series1.index)







# series2 = pd.Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
#
# print("Series: \n", series2)
# print("Series values: \n", series2.values)
# print("Series index: \n", series2.index)





# series2 = pd.Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
#
# print("Index d: ", series2['d'])
#
# series2['d'] = 6
# print("Updated index d: ", series2['d'])
#
# print("Series index range: \n", series2[['a', 'c', 'd']])






# series2 = pd.Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
#
# print("Numpy like filtering: ", series2[series2 > 0])
# print("Numpy like arithmetic: \n", series2 * 2)
# print("Pass series to Numpy function: \n", np.absolute(series2))






# series2 = pd.Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
#
# series2.name = 'Numbers'
# series2.index.name = 'Letters'
#
# print("Values title: ", series2.name)
# print("Index title: ", series2.index.name)





# data = {'state': ['Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
#         'year': [2001, 2002, 2001, 2002, 2003],
#         'pop': [1.7, 3.6, 2.4, 2.9, 3.2]}
# frame = pd.DataFrame(data)
#
# print(frame)
# print("First 5 rows: \n", frame.head())






# data = {'state': ['Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
#         'year': [2001, 2002, 2001, 2002, 2003],
#         'pop': [1.7, 3.6, 2.4, 2.9, 3.2]}
# frame = pd.DataFrame(data,
#                      columns=['year','state','pop'],
#                      index=['one', 'two', 'three', 'four', 'five'])
# print(frame)






# data = {'state': ['Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
#         'year': [2001, 2002, 2001, 2002, 2003],
#         'pop': [1.7, 3.6, 2.4, 2.9, 3.2]}
# frame = pd.DataFrame(data,
#                      columns=['year','state','pop'],
#                      index=['one', 'two', 'three', 'four', 'five'])
#
# print("Column retrieval: \n", frame['state'])
# print("Column retrieval: \n", frame.year)







# data = {'state': ['Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
#         'year': [2001, 2002, 2001, 2002, 2003],
#         'pop': [1.7, 3.6, 2.4, 2.9, 3.2]}
# frame = pd.DataFrame(data,
#                      columns=['year','state','pop'],
#                      index=['one', 'two', 'three', 'four', 'five'])
#
# print("Add and populate column:")
# frame['eastern'] = frame['state'] == 'Ohio'
# print(frame)
# print("Delete column:")
# del frame['eastern']
# print(frame)






# obj = pd.Series(np.arange(4.), index=['a', 'b', 'c', 'd'])
#
# print(obj)
# print("Index name: ", obj['b'])
# print("Index location: ", obj[1])
# print("Index multiple locations: \n", obj[[2, 3]])
# print("Index range: \n", obj[2:4])
# print("Index multiple names: \n", obj[['b', 'c']])
# print("Index condition: \n", obj[obj < 2])





#
# obj = pd.Series(np.arange(4.), index=['a', 'b', 'c', 'd'])
#
# print(obj['b':'c'])







# data = pd.DataFrame(np.arange(12).reshape((3,4)),
#                     index=['Colorado', 'Utah', 'New York'],
#                     columns=['one', 'two', 'three', 'four'])
#
# print(data)
# print(data[['two', 'four']])
# print(data[data['three'] > 5])








# data = pd.DataFrame(np.arange(12).reshape((3,4)),
#                     index=['Colorado', 'Utah', 'New York'],
#                     columns=['one', 'two', 'three', 'four'])
#
# print(data)
# print(data > 5)
#
# data[data > 5] = 0
# print(data)








# data = pd.DataFrame(np.arange(12).reshape((3,4)),
#                     index=['Colorado', 'Utah', 'New York'],
#                     columns=['one', 'two', 'three', 'four'])
#
# print("Index rows and columns by index:")
# print(data)
# print(data.iloc[2])
# print(data.iloc[2, [3, 0, 1]])
# print(data.iloc[[1, 2], [3, 0, 1]])







# data = pd.DataFrame(np.arange(12).reshape((3,4)),
#                     index=['Colorado', 'Utah', 'New York'],
#                     columns=['one', 'two', 'three', 'four'])
#
# print("Index rows and columns by index:")
# print(data)
# print(data.loc['Colorado'])
# print(data.loc['Colorado', ['two', 'three']])
# print(data.loc[['Colorado', 'Utah'], ['two', 'three']])






# data = pd.DataFrame(np.arange(12).reshape((3,4)),
#                     index=['Colorado', 'Utah', 'New York'],
#                     columns=['one', 'two', 'three', 'four'])
#
# print(data)
# print(np.add(data['one'], data['two']))






# series = pd.Series(range(4), index=['d', 'a', 'b', 'c'])
# print(series)
# print("Sort Series Ascending by index: \n", series.sort_index())
# print("Sort Series Descending by index: \n", series.sort_index(ascending=False))







# frame = pd.DataFrame({'b': [4, 7, -3, 2], 'a': [0, 1, 0, 1]})
# print(frame)
# print("Sort Series Ascending by value: \n",
#       frame.sort_values(by=['a', 'b']))
# print("Sort Series Descending by value: \n",
#       frame.sort_values(by=['a', 'b'], ascending=False))







# do the following: combine two series to form a dataFrame
# import pandas as pd
# import numpy as np

# ser1 = pd.Series(list('abcdefghijklmnopqrstuvwxyz'))
# ser2 = pd.Series(np.arange(26))
# data = {'col1': ser1,
#         'col2': ser2}
# frame = pd.DataFrame(data)
# print(frame.head())






# do the following:
# display 25th, 50th, 75th percentiles for a Series containing
# 100 random numbers from the standard normal distribution

# ser = pd.Series(np.random.randn(100))
# print(np.percentile(ser, q=[25, 50, 75]))

