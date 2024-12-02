import numpy as np
import pandas as pd


class Result:
    def __init__(self, accuracy, nearmiss, confusion):
        self.accuracy = accuracy
        self.nearmiss = nearmiss
        self.confusion = confusion

    def __str__(self):
        result = f'Accuracy: {self.accuracy}, Nearmiss: {self.nearmiss}\n'
        result += 'Confusion matrix:\n'
        result += str(self.confusion)
        return result


def train_test_split(df, test_size=0.2):
    mask = np.random.rand(len(df)) < 1 - test_size
    #  TODO: Select sets randomly basing on the users

    train_set = df[mask]
    test_set = df[~mask]

    return train_set, test_set


def make_random_batch(df, batch_size):
    return df.sample(n=batch_size)


def calculate_results(predictions_df, test_df, id_name='id', evaluation_value_name='rating'):
    return Result(calculate_accuracy(predictions_df, test_df, id_name, evaluation_value_name),
                  calculate_nearmiss(predictions_df, test_df, id_name, evaluation_value_name),
                  calculate_confusion(predictions_df, test_df, id_name, evaluation_value_name))


def calculate_accuracy(predictions_df, test_df, id_name='id', evaluation_value_name='rating'):
    correct = 0
    for _, prediction in predictions_df.iterrows():
        if prediction[evaluation_value_name] == \
                test_df[test_df[id_name] == prediction[id_name]][evaluation_value_name].values[0]:
            correct += 1
    return correct / len(predictions_df)


def calculate_nearmiss(predictions_df, test_df, id_name='id', evaluation_value_name='rating'):
    nearmiss = 0
    for _, prediction in predictions_df.iterrows():
        if abs(prediction[evaluation_value_name] -
               test_df[test_df[id_name] == prediction[id_name]][evaluation_value_name].values[0]) <= 1:
            nearmiss += 1
    return nearmiss / len(predictions_df)


def calculate_confusion(predictions_df, test_df, id_name='id', evaluation_value_name='rating'):
    predicted_list = []
    actual_list = []

    for _, prediction in predictions_df.iterrows():
        predicted_list.append(prediction[evaluation_value_name])

        actual = test_df[test_df[id_name] == prediction[id_name]]
        actual_list.append(actual[evaluation_value_name].values[0])

    predicted = pd.Series(predicted_list, name='Predicted')
    actual = pd.Series(actual_list, name='Actual')

    return pd.crosstab(predicted, actual)
