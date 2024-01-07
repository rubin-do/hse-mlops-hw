import pandas as pd
from dvc.api import open


def load_data(path, remote):
    with open(path, remote=remote) as f:
        df = pd.read_csv(f)
    return df
