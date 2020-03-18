import numpy as np
import pandas as pd


def load_data(country="Germany"):
    path = "thirdparty/COVID-19/csse_covid_19_data/csse_covid_19_time_series/"
    confirmed = pd.read_csv(path + "time_series_19-covid-Confirmed.csv")
    deaths = pd.read_csv(path + "time_series_19-covid-Deaths.csv")
    confirmed_country = confirmed[confirmed["Country/Region"] == country]
    deaths_country = deaths[deaths["Country/Region"] == country]
    confirmed_country = confirmed_country.iloc[:, 4:].to_numpy().flatten()
    deaths_country = deaths_country.iloc[:, 4:].to_numpy().flatten()
    data = pd.DataFrame(
        data=np.stack([confirmed_country, deaths_country], axis=1),
        columns=["confirmed", "deaths"],
    )
    return data


if __name__ == "__main__":
    load_data()
