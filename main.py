import numpy as np
import matplotlib.pyplot as plt

from functions import generate_data, prepare_data, save_data, get_weighted_sum, sigmoid, cross_entropy, update_bias, \
    update_weights

# save_data(data, 'dataset/titanic_data.txt')

bias = 0.5
l_rate = 0.0005
epochs = 150
epoch_loss = []

data, weights = prepare_data('dataset/titanic_original.csv')


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
        print('epoch ', e)
        print(average_loss)


# plot epoch_loss
plt.plot(np.arange(epochs), epoch_loss)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Epoch Loss')
plt.show()

train_model(data, weights, bias, l_rate, epochs)
