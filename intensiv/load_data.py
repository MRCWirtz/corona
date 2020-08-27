import pandas as pd
import requests
import datetime

url = 'https://www.divi.de/divi-intensivregister-tagesreport-archiv-csv/'

today = datetime.date.today()
formats = ["12-15", "12-15-2", "09-15", "09-15-2"]

while True:
    for f in formats:
        _id = '%s-%s' % (today.strftime('%Y-%m-%d'), f)
        r = requests.get(url + 'divi-intensivregister-%s/' % _id)
        value = r.text.split('value="')[1].split('"')[0]
        r = requests.get(url + 'divi-intensivregister-%s/viewdocument/%s' % (_id, value))
        if r.apparent_encoding == 'ascii':
            break
    try:
        data = pd.read_csv(url + 'divi-intensivregister-%s/viewdocument/%s' % (_id, value))
        if 'faelle_covid_aktuell' in data:
            print('(%s) Personen in Intensiv: ' % _id, data['faelle_covid_aktuell'].sum())
        else:
            break
    except pd.errors.ParserError:
        pass
    today -= datetime.timedelta(days=1)

# https://www.divi.de/divi-intensivregister-tagesreport-archiv-csv/divi-intensivregister-2020-08-26-12-15/viewdocument/5013
# https://www.divi.de/divi-intensivregister-tagesreport-archiv-csv/divi-intensivregister-2020-08-24-12-15/viewdocument/5008
# https://www.divi.de/divi-intensivregister-tagesreport-archiv-csv/divi-intensivregister-2020-06-26-12-15/viewdocument/3774
# https://www.divi.de/divi-intensivregister-tagesreport-archiv-csv/divi-intensivregister-2020-06-25-12-15-2/viewdocument/3919
# https://www.divi.de/divi-intensivregister-tagesreport-archiv-csv/divi-intensivregister-2020-05-19-09-15/viewdocument/3704
