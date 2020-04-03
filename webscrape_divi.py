import requests
import pandas as pd
import datetime
from bs4 import BeautifulSoup


url = "https://diviexchange.z6.web.core.windows.net/report.html"
resp = requests.get(url)
soup = BeautifulSoup(resp.text, "lxml")
table_header = soup.find("thead")
columns = [elem.text for elem in table_header.find("tr").find_all("th")][1:]
columns[9] = "Anzahl ECMO-Faelle pro Jahr"  # take correct encoding
indexname = table_header.find_all("tr")[1].find("th").text
rows = soup.find("tbody").find_all("tr")
indeces = [row.find("th").text for row in rows]
data = []
for row in rows:
    data.append([int(r.text) for r in row.find_all("td")])
df = pd.DataFrame(data, index=indeces, columns=columns)
df.index.name = indexname
df.to_csv("data_divi_{}.csv".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")))
