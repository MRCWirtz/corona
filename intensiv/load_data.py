import os
import numpy as np
import pandas as pd
import requests
import datetime
import matplotlib.pyplot as plt
from matplotlib import dates

cache, path = True, 'data/intensiv.csv'
url = 'https://www.divi.de/divi-intensivregister-tagesreport-archiv-csv/'

today = datetime.datetime.today()

if cache and os.path.exists(path):
    data = pd.read_csv(path, usecols=['date', 'faelle_covid_aktuell'])
    start_day = datetime.datetime.strptime(data['date'].iloc[-1], '%Y-%m-%d') + datetime.timedelta(days=1)
else:
    data = pd.DataFrame([], index=[], columns=['date', 'faelle_covid_aktuell'])
    start_day = datetime.datetime(2020, 4, 25)

formats = ["12-15", "12-15-2", "09-15", "09-15-2", "14-15", "14-15-2"]
split = 'href="/divi-intensivregister-tagesreport-archiv-csv/divi-intensivregister-'
r = [t[:34] for t in requests.get(url).text.split(split)]
while start_day <= today:
    day_str = start_day.strftime('%Y-%m-%d')
    try:
        _id, value = [t for t in r if t.startswith(day_str)][0].split('/viewdocument/')
    except IndexError:
        break
    try:
        print(url + 'divi-intensivregister-%s/viewdocument/%s' % (_id, value))
        _data = pd.read_csv(url + 'divi-intensivregister-%s/viewdocument/%s' % (_id, value))
        if ('faelle_covid_aktuell' in data) and (start_day.strftime('%Y-%m-%d') not in data['date'].values):
            data = data.append({'date': start_day.strftime('%Y-%m-%d'),
                                'datetime': start_day,
                                'faelle_covid_aktuell': _data['faelle_covid_aktuell'].sum()}, ignore_index=True)
            print('(%s) Personen in Intensiv: ' % _id, _data['faelle_covid_aktuell'].sum())
        else:
            break
    except pd.errors.ParserError:
        pass
    start_day += datetime.timedelta(days=1)

data['datetime'] = pd.Series([datetime.datetime.strptime(day, '%Y-%m-%d') for day in data['date']], index=data.index)
data.to_csv(path)
print(data)

plot_start = datetime.datetime.strptime('2020-06-01', '%Y-%m-%d')
mask = data['datetime'] >= plot_start
plt.plot_date(data['datetime'][mask], data['faelle_covid_aktuell'][mask], color='k', fmt='o')  # , ls='solid')
plt.text(0.5, 0.97, url, color='gray', transform=plt.gca().transAxes, size=9, horizontalalignment='center',
         verticalalignment='top')
# plt.gca().xaxis.set_minor_locator(dates.MonthLocator())
# plt.gca().xaxis.set_minor_formatter(dates.DateFormatter('%b'))
plt.gca().xaxis.set_major_locator(dates.MonthLocator())
plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%b'))
plt.ylabel('Covid in Intensivstation (DIVI)', color='k')
# plt.ylim(0, plt.gca().get_ylim()[1])
plt.grid(True)
plt.savefig('img/intensiv.png', bbox_inches='tight')
plt.close()
