import pandas as pd
import numpy as np
import os
from os import listdir

VARNAMES = ["HiTemp", "AvgTemp", "LowTemp", "HiDew", "AvgDew", "LowDew", "HiHumidity",
       "LowHumidity", "HiWind", "LoWind", "HiPressure", "LowPressue", "Precipitation"]


def read_data(directory=os.getcwd()):
    directory = []
    csv_files = listdir("data/")
    print(csv_files)
    for instance in csv_files:
        directory.append(load_from_file(instance))
    return directory


def load_from_file(file):
    file_name = "data/" + file
    with open(file_name, "r") as inputfile:
        month = inputfile.readline()

        # Setup some helper variables to get total length and
        # provide an array for the days of the month
        days = []
        ctr = 0

        # Rotate through the days and store in array
        line = inputfile.readline()
        while line[:3] != "Max":
            days.append(line.strip("\n"))
            line = inputfile.readline()
            ctr += 1

        # Begin to parse raw data
        data = []
        for i in range(6):
            # Reset categories on each iteration
            min = []
            avg = []
            max = []

            # Put each instance in it's own category
            for instance in range(ctr):
                split_array = []
                split_array = inputfile.readline().split()

                # Append if the value is not null
                if split_array[0] != "-":
                    max.append(float(split_array[0]))
                if split_array[1] != "-":
                    avg.append(float(split_array[1]))
                if split_array[2] != "-":
                    min.append(float(split_array[2]))

            # Append each category to data as long as it isn't empty
            if max:
                data.append(max)
            if avg:
                data.append(avg)
            if min:
                data.append(min)

            # Pass over the "Max, Avg, Min" line
            inputfile.readline()

    inputfile.close()
    return month, np.array(days), np.array(data)

