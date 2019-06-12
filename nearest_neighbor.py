import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
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

    weather_test = strat_test.drop("Label", axis=1)
    test_labels = strat_test["Label"].copy()

    # Setup a Nearest-Neighbor
    neigh2 = KNeighborsClassifier(n_neighbors=2)

    neigh3 = KNeighborsClassifier(n_neighbors=3)
    neigh5 = KNeighborsClassifier(n_neighbors=5)
    neigh10 = KNeighborsClassifier(n_neighbors=10)
    neigh2.fit(weather_train, weather_labels)
    neigh3.fit(weather_train, weather_labels)
    neigh5.fit(weather_train, weather_labels)
    neigh10.fit(weather_train, weather_labels)

    # Setup a Linear Model
    lin_reg = LinearRegression()
    lin_reg.fit(weather_train, weather_labels)

    # Do Predictions
    linear_predictions = lin_reg.predict(weather_test)
    neigh2_pred = neigh2.predict(weather_test)
    neigh3_pred = neigh3.predict(weather_test)
    neigh5_pred = neigh5.predict(weather_test)
    neigh10_pred = neigh10.predict(weather_test)

    # Print out errors
    lin_mae = mean_absolute_error(test_labels, linear_predictions)
    neigh2_mae = mean_absolute_error(test_labels, neigh2_pred)
    neigh3_mae = mean_absolute_error(test_labels, neigh3_pred)
    neigh5_mae = mean_absolute_error(test_labels, neigh5_pred)
    neigh10_mae = mean_absolute_error(test_labels, neigh10_pred)

    # Print Statements for accuracy scores
    print("Linear Accuracy Score: " + str(1 - lin_mae) + "%")
    print("NN 2 accuracy Score : " + str(1 - neigh2_mae) + "%")
    print("NN 3 accuracy Score : " + str(1 - neigh3_mae) + "%")
    print("NN 5 accuracy Score : " + str(1 - neigh5_mae) + "%")
    print("NN 10 accuracy Score : " + str(1 - neigh10_mae) + "%")


if __name__ == "__main__":
    main()
