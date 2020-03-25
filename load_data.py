import numpy as np
import pandas as pd


def load_data(country="Germany"):
    path = "thirdparty/covid-19/data/"
    data = pd.read_csv(path + "countries-aggregated.csv")
    data_country = data[data["Country"] == country].fillna(0)
    confirmed_country = data_country.Confirmed.to_numpy()
    deaths_country = data_country.Deaths.to_numpy()
    data = pd.DataFrame(
        data=np.stack([confirmed_country, deaths_country], axis=1),
        columns=["confirmed", "deaths"],
    )
    return data


if __name__ == "__main__":
    load_data()
