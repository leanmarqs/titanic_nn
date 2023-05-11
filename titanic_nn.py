import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from functions import prepare_data, reindex_dataframe, train_model, test_model, run

# create argument parser
parser = argparse.ArgumentParser(description='Train and test a logistic regression model on the Titanic dataset.')
parser.add_argument('-d', '--dataset', type=str, required=True, help='Path to the dataset file.')
parser.add_argument('-b', '--bias', type=float, default=0.5, help='Initial value for the bias term.')
parser.add_argument('-r', '--l_rate', type=float, default=0.0005, help='Learning rate for gradient descent.')
parser.add_argument('-e', '--epochs', type=int, default=2, help='Number of epochs to train the model for.')
args = parser.parse_args()

# prepare data
data, weights = prepare_data(args.dataset)

# split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2)

# this function puts data indices in correct crescent order

train_data = reindex_dataframe(train_data)
test_data = reindex_dataframe(test_data)

# train the model on the training set
# train_model(train_data, weights, args.bias, args.l_rate, args.epochs)

# test the model on the testing set
# test_model(test_data, weights, args.bias)

run(train_data, test_data, weights, args.bias, args.l_rate, args.epochs)
