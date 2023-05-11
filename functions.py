import datetime
import os

import pandas as pd
import numpy as np
import random as rg

from matplotlib import pyplot as plt

rg = np.random.default_rng()


# generate random data values.
def generate_data(n_features, n_values):
    features = rg.random((n_features, n_values))
    weights = rg.random((1, n_values))[0]
    targets = np.random.choice([0, 1], n_features)
    data = pd.DataFrame(features, columns=['x0', 'x1', 'x2'])
    data['targets'] = targets
    return data, weights


def prepare_data(input_file):
    data = pd.read_csv(input_file, usecols=['Survived', 'Pclass', 'Name', 'Sex', 'Age', 'Siblings/Spouses Aboard',
                                            'Parents/Children Aboard', 'Fare'])
    data['Sex'] = data['Sex'].map({'male': 1, 'female': 0})
    features = data[['Pclass', 'Sex', 'Age', 'Siblings/Spouses Aboard', 'Parents/Children Aboard', 'Fare']].values
    targets = data['Survived'].values
    values = data['Name'].values
    weights = rg.random((1, features.shape[1]))[0]
    data_final = pd.DataFrame(features,
                              columns=['Pclass', 'Sex', 'Age', 'Siblings/Spouses Aboard', 'Parents/Children Aboard',
                                       'Fare'])
    data_final['values'] = values
    data_final['targets'] = targets
    return data_final, weights


def save_data(data, output_file):
    with open(output_file, 'w') as f:
        f.write(data.to_string())


# the np.dot function works make a sum of each ((feature * weight) + bias)
def get_weighted_sum(feature, weights, bias):
    return np.dot(feature.astype(float), weights) + bias


# def sigmoid(w_sum):
#     return 1 / (1 + np.exp(-w_sum))

# the clip function maintain the w_sum size among -500 and 500, preventing overflow with a large w_sum to exp.
def sigmoid(w_sum):
    clipped_sum = np.clip(w_sum, -500, 500)
    return 1 / (1 + np.exp(-clipped_sum))


# def cross_entropy(target, prediction):
#     return -(target * np.log10(prediction) + (1 - target) * np.log10(1 - prediction))

# a small constant was added to prevent log10(0)
def cross_entropy(target, prediction):
    epsilon = 1e-10
    return -(target * np.log10(prediction + epsilon) + (1 - target) * np.log10(1 - prediction + epsilon))


def update_weights(weights, l_rate, target, prediction, feature):
    new_weights = []
    for x, w in zip(feature, weights):
        new_w = w + l_rate * (target - prediction) * x
        new_weights.append(new_w)
    return new_weights


def update_bias(bias, l_rate, target, prediction):
    return bias + l_rate * (target - prediction)


def run(train_data, test_data, weights, bias, l_rate, epochs):
    epoch_loss, final_bias, final_weights = train_model(train_data, weights, bias, l_rate, epochs)
    accuracy = test_model(test_data, final_weights, final_bias)

    save_results(epochs, epoch_loss, final_bias, l_rate, accuracy)


def save_results(epochs, epoch_loss, bias, l_rate, accuracy):
    # Create directory for current run
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    run_directory = f'results/run - {now}'
    os.makedirs(run_directory, exist_ok=True)

    # Plot epoch_loss
    plt.plot(epoch_loss, label=f"bias={bias:.10f}, l_rate={l_rate:.10f}")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Epoch Loss')
    plt.legend()
    # Save plot to file
    plot_filename = f'epoch_loss_{now}.png'
    plt.savefig(os.path.join(run_directory, plot_filename))
    plt.show()

    # Save accuracy to file
    accuracy_filename = f'accuracy_{now}.txt'
    with open(os.path.join(run_directory, accuracy_filename), 'w') as f:
        f.write(f'Accuracy: {accuracy}')


def train_model(data, weights, bias, l_rate, epochs):
    epoch_loss = []
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

    return epoch_loss, bias, weights


def test_model(data, weights, bias):
    correct = 0
    for i in range(len(data)):
        feature = data.loc[i][:-2]
        target = data.loc[i][-1]
        w_sum = get_weighted_sum(feature, weights, bias)
        prediction = sigmoid(w_sum)
        if prediction >= 0.5:
            if target == 1:
                correct += 1
        else:
            if target == 0:
                correct += 1

    accuracy = correct / len(data)
    print('**********************************')
    print('Accuracy:', accuracy)
    return accuracy


def reindex_dataframe(df):
    df.reset_index(drop=True, inplace=True)
    return df

