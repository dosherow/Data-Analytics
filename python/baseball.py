# structure for site

# baseballpress.com/lineups/date

import requests
from bs4 import BeautifulSoup
from datetime import date, timedelta
from time import time
from time import sleep
from random import randint
import pandas as pd

base_url = 'https://www.baseballpress.com/lineups/'
mydate = date(2019, 4, 15)

r = requests.get(base_url + mydate.strftime('%Y-%m-%d'))

soup = BeautifulSoup(r.text, 'html.parser')

chart = soup.find('div', {'class': 'row lineups'})


all_rows = []

for div in chart.find_all('div', {'class': 'col-md-6 col-xl-4 lineup-col'}):
    team = div.find('a', {'class': 'mlb-team-logo bc'}).find('div').text
    print(team)

    pitcher = div.find('div', {'class': 'col col--min player'}).find('a').text
    print(pitcher)

    lineup = div.find('div', {'class': 'col col--min'}).find('span', {'class': 'desktop-name'}).text
    print(lineup)
