import numpy as np
from copy import copy
import pandas as pd
import requests
import os
import datetime


def get_day_list(first, last):
    days = []
    i = 0
    while True:
        days.append(first + datetime.timedelta(days=i))
        i+=1
        if days[-1] == last:
            return days


def load_data(country="Germany"):
    path = "thirdparty/covid-19/data/"
    data = pd.read_csv(path + "countries-aggregated.csv")
    data_country = data[data["Country"] == country].fillna(0)
    confirmed_country = data_country.Confirmed.to_numpy()
    deaths_country = data_country.Deaths.to_numpy()
    data_country = data_country.Date.to_numpy()
    data = pd.DataFrame(
        data=np.stack([confirmed_country, deaths_country, data_country], axis=1),
        columns=["confirmed", "deaths", "dates"],
    )
    return data


def load_rki(raw=False):
    # Load data via REST api
    firsttime = True
    offset = 0
    while True:
        resp = requests.get("https://services7.arcgis.com/mOBPykOjAyBO2ZKk/arcgis/rest/services/RKI_COVID19/FeatureServer/0/query?where=1%3D1&outFields=Bundesland,Landkreis,Altersgruppe,Geschlecht,AnzahlFall,AnzahlTodesfall,Meldedatum,Datenstand,NeuerFall,NeuerTodesfall,Inzidenz&returnGeometry=false&outSR=4326&f=json&resultOffset={}".format(offset))
        if not resp.status_code == requests.codes.ok:
            raise RuntimeError("HTTP GET request failed")
        data_json = resp.json()
        features = data_json.get("features")
        if firsttime:
            data = pd.DataFrame([f.get("attributes") for f in features])
            firsttime = False
        else:
            data = data.append([f.get("attributes") for f in features], ignore_index=True)
        if data_json.get("exceededTransferLimit"):
            offset += 2000
        else:
            break
    data.loc[data.NeuerTodesfall == -9, "NeuerTodesfall"] = 0
    data.Meldedatum = pd.to_datetime(data.Meldedatum, unit="ms")
    if raw:
        return data
    pivot = pd.pivot_table(data, values=["AnzahlFall", "AnzahlTodesfall"], index="Meldedatum", aggfunc=np.sum)
    pivot = pivot.rename(columns={"AnzahlFall": "confirmed", "AnzahlTodesfall": "deaths"}, index={"Meldedatum": "dates"})
    daylist = get_day_list(pivot.index[0].to_pydatetime(), pivot.index[-1].to_pydatetime())
    complete_df = pd.DataFrame(index=pd.to_datetime(daylist), data={"confirmed": 0, "deaths": 0})
    complete_df.index.name = "Meldedatum"
    combined = pd.merge(pivot, complete_df, how="outer", on=["Meldedatum", "confirmed", "deaths"]).groupby("Meldedatum").sum()
    combined = combined.cumsum()
    return combined


if __name__ == "__main__":
    load_rki()
