from scipy.io import arff
import pandas as pd
import numpy as np

root_folder = "/Users/aniekmarkus/Documents/Git/_Projects/FIDisagreement"
datasets = ["iris", "vote", "compas", "german"]

data = datasets[3]

# Load data arff
input_data = arff.loadarff(root_folder + "/input-data/" + data + ".arff")
input_data = pd.DataFrame(input_data[0])

# Load data csv
input_data = pd.read_csv(root_folder + "/extras/data-openxai/" + data + ".csv")

# Check size
input_data.shape

# Necessary changes
if data == "compas":
    input_data.rename(columns={'two_year_recid': 'class'}, inplace=True)
elif data == "german":
    input_data.rename(columns={'credit.risk': 'class'}, inplace=True)

# Save data as csv
fileName = root_folder + "/input-data/" + data + ".csv"
input_data.to_csv(fileName, index=False)

# Change to arff in R
# data <- "compas"
# fileName <- paste0("/Users/aniekmarkus/Documents/Git/_Projects/FIDisagreement/input-data/", data)
# data <- read.csv(paste0(fileName,".csv"))
# data$class <- ifelse(data$class == 1, 0, 1) # switched labels for german (predict minority class)
# farff::writeARFF(data, paste0(fileName, ".arff"))