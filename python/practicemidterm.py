import numpy as np

#4.1

# take input of 3 numbers and output the largest odd number.
# if none are odd, show a message

# x = int(input("first no: "))
# y = int(input("second no: "))
# z = int(input("third no: "))
#
# my_list = sorted([int(x), int(y), int(z)])
#
# if x%2 == 0 and y%2 == 0 and z%2 == 0:
#     print("All are even")
# elif max(my_list)%2 != 0:
#     print(str(max(my_list)) + " is the biggest odd number")
# elif my_list[1]%2 !=0:
#     print(str(my_list[1]) + " is the biggest odd number")
# else:
#     print(str(my_list[0]) + " is the biggest odd number")








# data1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
# arry1 = np.array(data1)
# print("arry1 dim: ", arry1.ndim)
# print("arry1 shape: ", arry1.shape)
# print("arry1 type: ", arry1.dtype)
#
# print(" ")
#
# data2 = [[1, 2, 3], [4, 5, 6]]
# arry2 = np.array(data2)
# print("arry2 dim: ", arry2.ndim)
# print("arry2 shape: ", arry2.shape)
# print("arry2 type: ", arry2.dtype)








# data1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
# arry1 = np.array(data1)
# print("arry1 type: ", arry1.dtype)
#
# arry2 = arry1.astype(np.int64)
# print("arry2 type: ", arry2.dtype)

# data = [[1, 2, 3], [4, 5, 6]]
# arry = np.array(data)
#
# print("arry: \n", arry)
# print("arry + arry: \n", arry + arry)
# print("arry - arry: \n", arry * arry)
# print("1 / arry: \n", 1 / arry)
# print("arry ** arry: \n", arry ** arry)








# data1 = [[1, 2, 3], [4, 5, 6]]
# data2 = [[3, 2, 1], [6, 5, 4]]
# arry1 = np.array(data1)
# arry2 = np.array(data2)
#
# print("Comparisons between arrays of the same size yield boolean arrays: \n",
#       arry1 > arry2,
#       "\n\n",
#       arry1 == arry2)








# arry1 = np.arange(10)
# print("arry1: \n", arry1)
#
# print("arry1[5]: \n", arry1[5])
# print("arry1[5:8]: \n", arry1[5:8])
#






# arry1[5:8] = 12
# print("\nUpdate arry1[5:8] slice \n")
#
# print("arry1: \n", arry1)









# np_2d = np.array([[1, 3, 5], [2, 4, 6]])
#
# print(np_2d[0])
# print(np_2d[0][2])
# print(np_2d[0,2])
# print(np_2d[:,1:3])
# print(np_2d[1,:])






# names = np.array(['Bob', 'Joe', 'Will', 'Bob'])
#
# data = np.array([[1, 2, 3],
#                  [4, 5, 6],
#                  [7, 8, 9],
#                  [10, 11, 12]])
#
# print(names == 'Bob')
# print("Boolean Indexing: \n", data[names == 'Bob'])




# names = np.array(['Bob', 'Joe', 'Will', 'Bob'])
#
# data = np.array([[1, 2, 3],
#                  [4, 5, 6],
#                  [7, 8, 9],
#                  [10, 11, 12]])
#
# print("Boolean Indexing: \n", data[names != 'Bob'])






# names = np.array(['Bob', 'Joe', 'Will', 'Bob'])
#
# data = np.array([[1, 2, 3],
#                  [4, 5, 6],
#                  [7, 8, 9],
#                  [10, 11, 12]])
#
# print("Boolean Indexing: \n", data[names == 'Bob', :2])






# names = np.array(['Bob', 'Joe', 'Will', 'Bob'])
#
# data = np.array([[1, 2, 3],
#                  [4, 5, 6],
#                  [7, 8, 9],
#                  [10, 11, 12]])
#
# print("Boolean Indexing: \n", data[(names == 'Bob') | (names == 'Joe')])
# print("Boolean Indexing: \n", data[(names == 'Bob') & (names == 'Joe')])








# print("Random integers: \n",
#       np.random.randint(low=1, high=100, size=4))
# print("Random integers: \n",
#       np.random.randint(low=1, high=100, size=(4,3)))
# print("Random from standard normal dist: \n",
#       np.random.randn(4))
# print("Random from standard normal dist: \n",
#       np.random.randn(4, 3))



# calculate the mean for the first column from data
# for all rows corresponding to Bob. the answer is 5.5
# names = np.array(['Bob', 'Joe', 'Will', 'Bob'])
#
# data = np.array([[1, 2, 3],
#                  [4, 5, 6],
#                  [7, 8, 9],
#                  [10, 11, 12]])
#
# print(np.mean(data[names == 'Bob', 0]))






# drop all the missing values from array below
# and save new array as integers in a variable 'b'

# a = np.array([1,2,3,np.nan,5,6,7,np.nan])
# b = a[np.isnan(a) == False].astype(np.int64)
#
# print(b)





