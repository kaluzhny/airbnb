import pandas as pd
import numpy as np


def read_from_csv(path, seed, max_rows=None):
    np.random.seed(seed)
    df = pd.read_csv(path)
    df = df.reindex(np.random.permutation(df.index))
    if max_rows is not None:
        df = df.head(max_rows)
    return df
