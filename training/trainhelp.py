import numpy as np


def train_test_split(df, test_size=0.2):

    mask = np.random.rand(len(df)) < 1 - test_size
    #  TODO: Select sets randomly basing on the users

    train_set = df[mask]
    test_set = df[~mask]

    return train_set, test_set


def make_random_batch(df, batch_size):
    return df.sample(n=batch_size)


def calculate_accuracy(predictions_df, test_df, id_name='id', evaluation_value_name='rating'):
    correct = 0
    for _, prediction in predictions_df.iterrows():
        if prediction[evaluation_value_name] == test_df[test_df[id_name] == prediction[id_name]][evaluation_value_name].values[0]:
            correct += 1
    return correct / len(predictions_df)

