import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date

# now = datetime.now()
# print("Timestamp: year, month, day, hour, minute, second, timezone \n",
#       now)
#
# print("Year: ", now.year)
# print("Month: ", now.month)
# print("Day: ", now.day)
#
# #find elapsed time between two timestamps
# delta = datetime(2019, 2, 11) - datetime(2008, 9, 3, 1, 15)
# print("Difference in days: ", delta.days)
# print("Difference in seconds: ", delta.seconds)










# stamp = datetime(2018, 10, 30)
# print("Date as string: ", str(stamp))
# print("Date as formatted string: ", stamp.strftime('%Y-%m-%d'))
#
# value = '2018-10-30'
# print("String to date: ", datetime.strptime(value, '%Y-%m-%d'))







# dates = [datetime(2019, 1, 2),
#          datetime(2019, 1, 5),
#          datetime(2019, 1, 7),
#          datetime(2019, 1, 8),
#          datetime(2019, 1, 10),
#          datetime(2019, 1, 12)]
#
# ts = pd.Series(np.random.randn(6), index=dates)
# print("Use timestamp as Series index: ", ts)











# dates = [datetime(2019, 1, 2),
#          datetime(2019, 1, 2),
#          datetime(2019, 1, 7),
#          datetime(2019, 1, 8),
#          datetime(2019, 1, 10),
#          datetime(2019, 1, 12)]
#
# ts = pd.Series(np.random.randn(6), index=dates)
# print("Retreive RN for 1/10/2019: ", ts['1/10/2019'])
# print("Retrieve RN for 1/10/2019: ", ts['20190110'])
#
# print("Retrieve RN for 1/2/2019: \n", ts['20190102'])








# dates2 = pd.date_range('1/1/2000', periods=1000)
#
# ts2 = pd.Series(np.random.randn(1000), index=dates2)
#
# print("Retrieve RN for 2001: \n", ts2['2001'])
# print("Retrieve RN for 05/2001: \n", ts2['2001-05'])
# print("Retrieve RN for 01/06/2001 to 01/11/2001: \n",
#       ts2['1/6/2001':'1/11/2001'])








# dates3 = pd.date_range('1/1/2000', periods=100)
#
# ts3 = pd.DataFrame(np.random.randn(100, 4),
#                    index=dates3,
#                    columns=['Colorado', 'Texas', 'New York', 'Ohio'])
#
# print("Retrieve RN for 2001: \n", ts3.loc['1-2000'])







#
# print("Date range with start date and end date for hourly frequency: ",
#       pd.date_range('2000-01-01', '2000-01-02', freq='H'))







# write a program that takes your birthdate as input and outputs your age in days

# print("Please enter your birthday")
# bd_y = input("Year: ")
# bd_m = input("Month (1-12): ")
# bd_d = input("Day: ")
#
# now = date.today()
#
# birthdate = date(int(bd_y), int(bd_m), int(bd_d))
#
# age = now - birthdate
#
# print("You are", age.days, "days old")
