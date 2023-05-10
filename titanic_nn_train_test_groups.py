import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from functions import generate_data, prepare_data, save_data, get_weighted_sum, sigmoid, cross_entropy, update_bias, \
    update_weights, test_model, plot_data, reindex_dataframe

# save_data(data, 'dataset/titanic_data.txt')

bias = 0.5
l_rate = 0.0005
epochs = 200
epoch_loss = []

# prepare data.
data, weights = prepare_data('dataset/titanic_original.csv')

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2)

train_data = reindex_dataframe(train_data)
test_data = reindex_dataframe(test_data)


def train_model(data, weights, bias, l_rate, epochs):
    for e in range(epochs):
        individual_loss = []
        for i in range(len(data)):
            # the [:-2] select the first 6 cols of the table data, where we find the features.
            feature = data.loc[i][:-2]
            # the [-1] select the very last col of the table data, where we find the targets.
            target = data.loc[i][-1]
            w_sum = get_weighted_sum(feature, weights, bias)
            prediction = sigmoid(w_sum)
            loss = cross_entropy(target, prediction)
            individual_loss.append(loss)
            # gradient descent
            weights = update_weights(weights, l_rate, target, prediction, feature)
            bias = update_bias(bias, l_rate, target, prediction)
        average_loss = sum(individual_loss) / len(individual_loss)
        epoch_loss.append(average_loss)
        print('**********************************')
        print('Epoch:', e)
        print('Average Loss:', average_loss)


# Train the model on the training set
train_model(train_data, weights, bias, l_rate, epochs)
test_model(test_data, weights, bias)

# plot epoch_loss
plot_data(epochs, epoch_loss)
