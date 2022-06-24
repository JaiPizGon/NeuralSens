import pkg_resources
import pandas as pd


def load_daily_data_demand_tr():
    """Return a dataframe with values of temperature and working day to predict electrical demand.

    Contains the following fields:
        DATE         1980 non-null datetime : date of the measure
        DEM          1980 non-null float    : electrical demand
        WD           1980 non-null float    : Working Day Index
        TEMP         1980 non-null float    : weather temperature
    """
    # This is a stream-like object. If you want the actual info, call
    # stream.read()
    stream = pkg_resources.resource_stream(__name__, 'data/daily_demand_tr.csv')
    return pd.read_csv(stream, sep=";", index_col="DATE")

def load_daily_data_demand_tv():
    """Return a dataframe with values of temperature and working day to predict electrical demand.

    Contains the following fields:
        DATE         1980 non-null datetime : date of the measure
        DEM          1980 non-null float    : electrical demand
        WD           1980 non-null float    : Working Day Index
    """
    # This is a stream-like object. If you want the actual info, call
    # stream.read()
    stream = pkg_resources.resource_stream(__name__, 'data/daily_demand_tv.csv')
    return pd.read_csv(stream, sep=";", index_col="DATE")
