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
    start_day = datetime.datetime.strptime(data['date'].iloc[-1], '%Y-%m-%d')
else:
    data = pd.DataFrame([], index=[], columns=['date', 'faelle_covid_aktuell'])
    start_day = datetime.datetime(2020, 4, 25)

formats = ["12-15", "12-15-2", "09-15", "09-15-2", "14-15", "14-15-2"]
while start_day <= today:
    for f in formats:
        _id = '%s-%s' % (start_day.strftime('%Y-%m-%d'), f)
        r = requests.get(url + 'divi-intensivregister-%s/' % _id)
        value = r.text.split('value="')[1].split('"')[0]
        r = requests.get(url + 'divi-intensivregister-%s/viewdocument/%s' % (_id, value))
        if r.apparent_encoding == 'ascii':
            break
    try:
        _data = pd.read_csv(url + 'divi-intensivregister-%s/viewdocument/%s' % (_id, value))
        if ('faelle_covid_aktuell' in data) and (start_day.strftime('%Y-%m-%d') not in data['date'].values):
            data = data.append({'date': start_day.strftime('%Y-%m-%d'),
                                'faelle_covid_aktuell': _data['faelle_covid_aktuell'].sum()}, ignore_index=True)
            print('(%s) Personen in Intensiv: ' % _id, _data['faelle_covid_aktuell'].sum())
        else:
            break
    except pd.errors.ParserError:
        pass
    start_day += datetime.timedelta(days=1)

data.to_csv(path)
print(data)

plt.plot_date(data['date'], data['faelle_covid_aktuell'], marker='o', color='k')
plt.gca().xaxis.set_minor_locator(dates.MonthLocator())
plt.gca().xaxis.set_minor_formatter(dates.DateFormatter('%b'))
plt.gca().xaxis.set_major_locator(dates.YearLocator())
plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%Y'))
plt.ylabel('Covid in Intensivstation (DIVI)', color='k')
plt.grid(True)
plt.savefig('img/intensiv.png', bbox_inches='tight')
plt.close()
