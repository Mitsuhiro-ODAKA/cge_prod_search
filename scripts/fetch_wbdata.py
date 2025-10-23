import wbdata
import pandas as pd
import datetime
import yaml


def fetch_worldbank_data(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    countries = config["countries"]
    indicators = config["indicators"]

    start_date = datetime.datetime.strptime(config["date_range"]["start"], "%Y-%m-%d")
    end_date = datetime.datetime.strptime(config["date_range"]["end"], "%Y-%m-%d")
    data_date = (start_date, end_date)

    print("Fetching data from World Bank...")
    df = wbdata.get_dataframe(indicators, country=countries, date=data_date)

    df = df.reset_index()
    df = df.rename(columns={"date": "Year", "country": "Country"})
    df = df.dropna()
    df = df.sort_values(["Country", "Year"]).reset_index(drop=True)

    output_path = config["output_csv"]
    df.to_csv(output_path, index=False)
    print(f"Data saved to: {output_path}")

    return df
