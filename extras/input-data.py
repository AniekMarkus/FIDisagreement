from scipy.io import arff
import pandas as pd
import numpy as np

root_folder = "..."
datasets = ["iris", "vote", "compas", "german", "copdmortality", "heartfailurestroke"]

data = datasets[0]

# Load data arff
input_data = arff.loadarff(root_folder + "/input-data/" + data + ".arff")
input_data = pd.DataFrame(input_data[0])

# Load data csv
input_data = pd.read_csv(root_folder + "/extras/data-openxai/" + data + ".csv")

# Check size
input_data.shape

# Check rate
input_data['class'].sum() / len(input_data['class']) * 100.0

# Necessary changes
if data == "compas":
    input_data.rename(columns={'two_year_recid': 'class'}, inplace=True)
elif data == "german":
    input_data.rename(columns={'credit.risk': 'class'}, inplace=True)

# Save data as csv
fileName = root_folder + "/input-data/" + data + ".csv"
input_data.to_csv(fileName, index=False)

# Import german credit data
# german_credit <- read.table("http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data-numeric")

# colnames(german_credit) <- c("chk_acct", "duration", "credit_his", "purpose", 
#                            "amount", "saving_acct", "present_emp", "installment_rate", "sex", "other_debtor", 
#                            "present_resid", "property", "age", "other_install", "housing", "n_credits", 
#                            "job", "n_people", "telephone", "foreign", "class")

# colnames(german_credit) <-c(colnames(german_credit)[1:24], "class")
# german_credit[,"class"] <- ifelse(german_credit[,"class"] == 1, 0, 1) # 1 = Good, 2 = Bad

# Change to arff in R
# data <- "german"
# fileName <- paste0("/Users/aniekmarkus/Documents/Git/_Projects/FIDisagreement/input-data/", data)
# data <- read.csv(paste0(fileName,".csv"))
# farff::writeARFF(data, paste0(fileName, ".arff"))

