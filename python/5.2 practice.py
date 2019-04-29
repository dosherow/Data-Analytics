import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# write a program that lets you guess what number python is thinking

# number = np.random.randint(1, 10)
# print('Guess the number between 1 and 10 Im thinking of.')
#
# guessesTaken = 0
#
# while guessesTaken >= 0:
#
#     print('Take a guess: ')
#     guess = input()
#     guess = int(guess)
#
#     guessesTaken = guessesTaken + 1
#
#     if guess < number:
#         print('Your guess is too low')
#
#     if guess > number:
#         print('Your guess is too high')
#
#     if guess == number:
#         break
#
# print('You guessed my number', number, 'correctly!')
# print("It took you", guessesTaken, 'guesses!')









# display these blank subplots using subplot function

# plt.subplot(2, 1, 1)
# plt.xticks(())
# plt.yticks(())
#
# plt.subplot(2, 3, 4)
# plt.xticks(())
# plt.yticks(())
#
# plt.subplot(2, 3, 5)
# plt.xticks(())
# plt.yticks(())
#
# plt.subplot(2, 3, 6)
# plt.xticks(())
# plt.yticks(())
#
# plt.show()










# create two dataframe variables containing the data
# os.chdir('/Users/arpi-admin/Documents/ITP449_Files')
#
# df_citiBike0218 = pd.read_csv('citiBike0218.csv')
# df_citiBike0718 = pd.read_csv('citiBike0718.csv')
# pd.set_option('display.max_columns', None)
#
# print(df_citiBike0218.head())
# print(df_citiBike0718.head())










# # create two ndarray variables for trip duration in minutes
# os.chdir('/Users/arpi-admin/Documents/ITP449_Files')
#
# df_citiBike0218 = pd.read_csv('citiBike0218.csv')
# df_citiBike0718 = pd.read_csv('citiBike0718.csv')
# pd.set_option('display.max_columns', None)
#
# # trip duration in minutes variables
# ar_tripDuration0218 = df_citiBike0218['tripduration'].values/60
# ar_tripDuration0718 = df_citiBike0718['tripduration'].values/60








# create a figure of 2 x 1 subplots that displays the histograms of trip durations
# fig = plt.figure()
# ax1 = fig.add_subplot(2, 1, 1)
# ax2 = fig.add_subplot(2, 1, 2)
#
# ax1.hist(ar_tripDuration0218, color='b', alpha=.8)
# ax1.set_title('02/18 Trip Duration (in minutes)')
#
# ax2.hist(ar_tripDuration0718, color='m', alpha=.8)
# ax2.set_title('07/18 Trip Duration (in minutes)')
#
# plt.show()










# scale the histograms to display more details
# ax1.hist(ar_tripDuration0218, range = (0, 50), bins = 25, color='b', alpha=.8)








# create a 2x1 subplots that displays the histograms of the trip durations for subscribers and customers
# fig = plt.figure()
# ax1 = fig.add_subplot(2, 1, 1)
# ax2 = fig.add_subplot(2, 1, 2)
#
# tripDuration0218Sub = []
# tripDuration0218Cus = []
#
# # separate duration of subscribers and customers
# ar_userType0218 = df_citiBike0218['usertype'].values
#
# for i in range(len(ar_userType0218)):
#     if ar_userType0218[i] == 'Subscriber':
#         tripDuration0218Sub.append(ar_tripDuration0218[i])
#     else:
#         tripDuration0218Cus.append(ar_tripDuration0218[i])
# ax2.set_ylim([0, 10500])
#
# plt.show()