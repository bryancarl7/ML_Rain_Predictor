import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression, LogisticRegressionCV
from sklearn.tree import DecisionTreeRegressor
from loader import read_data

VARNAMES = ["HiTemp", "AvgTemp", "LowTemp", "HiDew", "AvgDew", "LowDew", "HiHumidity",
            "LowHumidity", "HiWind", "LoWind", "HiPressure", "LowPressure", "Precipitation"]

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

    # Create a stratified shuffle split for the data
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

    # Setup a Linear Regression Model
    lin_reg = LinearRegression()
    lin_reg.fit(weather_train, weather_labels)

    # Setup a Logistic Regression model
    log_reg = LogisticRegressionCV(cv=5, random_state=0, multi_class='multinomial', )
    log_reg.fit(weather_train, weather_labels)

    # Do Predictions
    linear_predictions = lin_reg.predict(weather_train)
    linear_test_predictions = lin_reg.predict(weather_test)
    logistic_predictions = log_reg.predict(weather_train)
    logistic_test_predictions = log_reg.predict(weather_test)

    lin_mae = mean_absolute_error(weather_labels, linear_predictions)
    linear_test_mae = mean_squared_error(test_labels, linear_test_predictions)
    log_mae = mean_absolute_error(weather_labels, logistic_predictions)
    logistic_test_mae = mean_squared_error(test_labels, logistic_test_predictions)
    print("Linear MeanAbsoluteError on training data  : " + str(1 - lin_mae) + "%")
    print("Linear MeanAbsoluteError on test data      : " + str(1 - linear_test_mae) + "%")
    print("Logistic MeanAbsoluteError on training data: " + str(1 - log_mae) + "%")
    print("Logistic MeanAbsoluteError on test data    : " + str(1 - logistic_test_mae) + "%")


if __name__ == "__main__":
    main()
