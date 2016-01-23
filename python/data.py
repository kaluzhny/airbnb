import pandas as pd
import numpy as np


def read_from_csv(path, seed, max_rows=None):
    df = pd.read_csv(path)
    np.random.seed(seed)
    # df = df.reindex(np.random.permutation(df.index))
    if max_rows is not None:
        df = df.head(max_rows)
    return df
