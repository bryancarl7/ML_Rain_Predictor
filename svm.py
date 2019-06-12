import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn import svm
from loader import read_data

VARNAMES = ["HiTemp", "AvgTemp", "LowTemp", "HiDew", "AvgDew", "LowDew", "HiHumidity",
            "LowHumidity", "HiWind", "LoWind", "HiPressure", "LowPressure", "Precipitation"]


# I Followed this article on CV_grid search
# https://medium.com/@aneesha/svm-parameter-tuning-in-scikit-learn-using-gridsearchcv-2413c02125a0
# incredibly enlightening. I hope this is ok, if not please let me know and I will
# remove this code from the project
def best_parameters(data, labels, folds):
    lr = [0.001, 0.01, 0.1, 1, 10]
    gamma = [0.001, 0.01, 0.1, 1]
    kernel = ['linear', 'rbf']
    grid = {'kernel': kernel, 'C': lr, 'gamma' : gamma}
    grid_search = GridSearchCV(svm.SVC(kernel='rbf'), grid, cv=folds)
    grid_search.fit(data, labels)
    print(grid_search.best_params_)
    return grid_search.best_params_


def main():
    data = read_data()
    data_frames = []
    ctr = 0
    length = len(data)

    # Continue through the list until there is nothing left
    while ctr < length:
        data_frames.append(pd.DataFrame(np.transpose(data[ctr][2])))
        ctr += 1

    print("\n" + str(ctr) + " : Months Processed\n")

    # Setup the final datafram with the columns
    df = pd.concat(data_frames, ignore_index=True)
    df.columns = VARNAMES

    # Add in Label Column and make them 1 or 0
    df["Label"] = df["Precipitation"] > 0
    df.Label = df.Label.astype(int)
    df = df.drop(columns=["Precipitation"])
    print(df)

    # Get rid of precipitation
    correlation_matrix = df.corr()
    print(correlation_matrix["Label"].sort_values(ascending=False))

    # Create a stratified shuffle split for the data 80/20 for train/test
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(df, df["Label"]):
        strat_train = df.loc[train_index]
        strat_test = df.loc[test_index]

    # Prepare the Data for Machine Learning
    weather_train = strat_train.drop("Label", axis=1)
    weather_labels = strat_train["Label"].copy()

    # Setup test set
    weather_test = strat_test.drop("Label", axis=1)
    test_labels = strat_test["Label"].copy()

    # Setup a Linear Regression model for comparison
    lin_reg = LinearRegression()
    lin_reg.fit(weather_train, weather_labels)

    # Setup a Grid Searched SVM
    params = best_parameters(weather_train, weather_labels, 5)
    support_vector = svm.SVC(kernel=params['kernel'], C=params['C'], gamma=params['gamma'])
    support_vector.fit(weather_train, weather_labels)

    # Do Linear Predictions
    linear_predictions = lin_reg.predict(weather_train)
    linear_test_predictions = lin_reg.predict(weather_test)

    # Do SVM Predictions
    svm_predictions = support_vector.predict(weather_train)
    svm_test_predictions = support_vector.predict(weather_test)

    # Print out errors
    lin_mae = mean_absolute_error(weather_labels, linear_predictions)
    lin_test_mae = mean_absolute_error(test_labels, linear_test_predictions)
    svm_mae = mean_absolute_error(weather_labels, svm_predictions)
    svm_test_mae = mean_absolute_error(test_labels, svm_test_predictions)
    print("Linear MeanAbsoluteError on train: " + str(1 - lin_mae) + "%")
    print("Linear MeanAbsoluteError on test : " + str(1 - lin_test_mae) + "%")
    print("SVM MeanAbsoluteError on train   : " + str(1 - svm_mae) + "%")
    print("SVM MeanAbsoluteError on test    : " + str(1 - svm_test_mae) + "%")


if __name__ == "__main__":
    main()
