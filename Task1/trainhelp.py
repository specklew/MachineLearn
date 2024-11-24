import pandas as pd
import numpy as np


def train_test_split(df, test_size=0.2):

    df = df.sample(frac=1)

    mask = np.random.rand(len(df)) < 1 - test_size

    train_set = df[mask]
    test_set = df[~mask]

    return train_set, test_set

