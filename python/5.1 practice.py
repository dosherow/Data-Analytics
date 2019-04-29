import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# do the following:
# write a program that asks the user how many fibonacci numbers
# to generate and tehn generates them.
# fibonacci sequence is a sequence of numbers where the next number in the sequence
# is the sum of previous 2 numbers in sequence. the sequence looks like
# this: 1, 1, 2, 3, 5, 8, 13...

# count = int(input("How many fibonacci numbers would you like?: "))
# i = 1
#
# if count == 0:
#     fib = []
# elif count == 1:
#     fib = [1]
# elif count == 2:
#     fib = [1, 1]
# elif count > 2:
#     fib = [1, 1]
#     while i < (count - 1):
#         fib.append(fib[i] + fib[i - 1])
#         i += 1
#
# print(fib)
#
# def FibNum(n):
#     if n<=1:
#         return n
#     else:
#         return (FibNum(n-1)+FibNum(n-2))
#
# nterms = int(input("How many Fibonacci terms should I calculate?"))
# for i in range(nterms+1):
#     if i==0:
#         pass
#     else:
#         print(FibNum(i))








# data = np.arange(10)
# plt.plot(data)
#
# plt.show()




#
# fig = plt.figure()
# ax1 = fig.add_subplot(2, 2, 1)
# ax2 = fig.add_subplot(2, 2, 2)
# ax3 = fig.add_subplot(2, 2, 3)
#
# plt.show()







# fig = plt.figure()
# ax1 = fig.add_subplot(2, 2, 1)
# ax2 = fig.add_subplot(2, 2, 2)
# ax3 = fig.add_subplot(2, 2, 3)
#
# ax1.hist(np.random.randn(100),
#          bins=20,
#          color='m')
# ax2.scatter(np.arange(30),
#             np.arange(30) + 3 * np.random.randn(30),
#             color='b')
# ax3.plot(np.random.randn(50).cumsum(),
#          'g--')
# plt.show()









# fig, axes = plt.subplots(1, 4, sharey=True)
#
# data = np.random.randn(500)
# axes[0, ].hist(data, bins=10, color='#f4d942')
# axes[1, ].hist(data, bins=30, color='#f4d942')
# axes[2, ].hist(data, bins=50, color='#f4d942')
# axes[3, ].hist(data, bins=70, color='#f4d942')
#
# plt.show()








# data = np.random.randn(30)
# plt.plot(data,
#          color='k',
#          linestyle='--',
#          marker='o')
#
# #shortcut for the same plot
# plt.plot(data, 'ko--')
#
# plt.show()








# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
#
# ax.plot(np.random.randn(1000).cumsum())
#
# plt.show()









# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
#
# ax.plot(np.random.randn(1000).cumsum())
# ticks = ax.set_xticks([0, 250, 500, 750, 1000])
# labels = ax.set_xticklabels(['one', 'two', 'three', 'four', 'five'],
#                              rotation=30,
#                              fontsize='small')
# ax.set_title('My first plot')
# ax.set_xlabel('Stage')
# plt.show()






#
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
#
# ax.plot(np.random.randn(1000).cumsum(), 'k', label='one')
# ax.plot(np.random.randn(1000).cumsum(), 'k--', label='two')
# ax.plot(np.random.randn(1000).cumsum(), 'k.', label='three')
# ax.legend(loc='best')
# plt.show()








# display a 3 x 3 blank subplot
# plt.subplot(2, 3, 1)
# plt.subplot(2, 3, 2)
# plt.subplot(2, 3, 3)
# plt.subplot(2, 3, 4)
# plt.subplot(2, 3, 5)
# plt.subplot(2, 3, 6)
# plt.show()








# do following scaterrplot

# x = np.random.randint(1, 100, 50)
# y = np.random.randint(1, 100, 50)
# pltColor = x - y
# plt.scatter(x, y, c=pltColor, s=100, alpha=.5, marker='o')
# plt.show()