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


def load_jhu(country="Germany"):
    """ Get data from Johns Hopkins University (CSSE) over thirdparty Github repository """
    path = "thirdparty/covid-19/data/"
    data = pd.read_csv(path + "countries-aggregated.csv")
    data_country = data[data["Country"] == country].fillna(0)
    confirmed_country = data_country.Confirmed.to_numpy()
    deaths_country = data_country.Deaths.to_numpy()
    data = pd.DataFrame(
        index=pd.to_datetime(data_country.Date),
        data=np.stack([confirmed_country, deaths_country], axis=1),
        columns=["confirmed", "deaths"],
    )
    return data


def load_rki(raw=False):
    """ Get data from Robert Koch Institut (unprepared) """
    # Load data via REST api
    offset = 0
    data = pd.DataFrame()
    while True:
        resp = requests.get("https://services7.arcgis.com/mOBPykOjAyBO2ZKk/arcgis/rest/services/RKI_COVID19/FeatureServer/0/query?where=1%3D1&outFields=Bundesland,Landkreis,Altersgruppe,Geschlecht,AnzahlFall,AnzahlTodesfall,Meldedatum,Datenstand,NeuerFall,NeuerTodesfall&returnGeometry=false&outSR=4326&f=json&resultOffset={}".format(offset))
        if not resp.status_code == requests.codes.ok:
            raise RuntimeError("HTTP GET request failed")
        data_json = resp.json()
        features = data_json.get("features")
        data = data.append([f.get("attributes") for f in features], ignore_index=True)
        if data_json.get("exceededTransferLimit"):
            offset += 2000
        else:
            break
    filename = "raw_rki_data_{}.csv".format(pd.to_datetime(data.Datenstand[0]).strftime("%Y-%m-%d"))
    data.to_csv(filename)
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
