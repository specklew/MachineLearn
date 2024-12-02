from training.trainhelp import train_test_split, make_random_batch, calculate_results
import pandas as pd


class Classifier:
    def __init__(self, batch_size: int = 1024, test_count: int = 3, test_divide: float = 0.2):
        self.batch_size = batch_size
        self.test_count = test_count
        self.test_divide = test_divide

    def fit_test_predict(self, x_train: pd.DataFrame) -> float:

        sum_accuracy = 0

        print("\n")
        print("=====================")
        print("    Starting test    ")
        print("---------------------")
        print("TEST COUNT:", self.test_count)
        print("TEST DIVIDE:", self.test_divide)
        print("BATCH SIZE:", self.batch_size)
        print("---------------------")
        print("\n")

        for test_num in range(self.test_count):

            test_name = "--- Test" + str(test_num + 1) + " ---"
            print(test_name)
            train = make_random_batch(x_train, self.batch_size)

            train, test = train_test_split(train, test_size=self.test_divide)

            predictions = self.fit_predict(train, test)
            result = calculate_results(predictions, test)
            print(result)
            print("-" * len(test_name) + "\n")

            sum_accuracy += result.accuracy

        avg_accuracy = sum_accuracy / self.test_count

        print("\n")
        print("=====================")
        print("    Test finished    ")
        print("---------------------")
        print("AVG ACCURACY:", round(avg_accuracy, 4))
        print("---------------------")

        return avg_accuracy

    def fit_predict(self, x_train: pd.DataFrame, x_test: pd.DataFrame) -> pd.DataFrame:
        pass

