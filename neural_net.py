import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
import tensorflow as tf
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

    # A couple of statements to get rid of verbosity (only shows errors, not warnings)
    old_v = tf.logging.get_verbosity()
    tf.logging.set_verbosity(tf.logging.ERROR) # old_v can be used to set_verbosity(old_v) back to old messages

    # Setup the inputs
    nn_inputs = 28 * 28
    nn_hidden = 16
    nn_outputs = 10

    # Setup variables
    X = tf.placeholder(tf.float32, shape=(None, nn_inputs), name="X")
    y = tf.placeholder(tf.int64, shape=(None), name="y")

    # Building the tf.name_scope
    learning_rate = 0.01

    # Implemented neural net using sigmoid function, found in activation parameter of hidden_1
    with tf.name_scope("dnn"):
        hidden_2 = tf.layers.dense(X, nn_hidden, name="hidden_2", reuse=tf.AUTO_REUSE, activation=tf.nn.sigmoid)
        logits = tf.layers.dense(hidden_2, nn_outputs, reuse=tf.AUTO_REUSE, name="outputs")

    # Created loss function
    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")

    # Implemented a gradient descent
    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)

    # Evaluation Functions
    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # Set Epoch and Batch Size
    epochs = 40
    batch_size = 50

    # Begin evaluation
    with tf.Session() as sess:
        init.run()
        for epoch in range(epochs):
            sess.run(training_op, feed_dict={X: weather_train, y: weather_labels})

            # Report the training accuracies
            training_acc = accuracy.eval(feed_dict={X: weather_train, y: weather_labels})
            validation_acc = accuracy.eval(feed_dict={X: weather_test,
                                                      y: test_labels})

            print("Epoch :" + str(epoch) + "\n[Train Accuracy] : " + str(training_acc)
                  + "\n[Validation Accuracy] :" + str(validation_acc))


if __name__ == "__main__":
    main()
