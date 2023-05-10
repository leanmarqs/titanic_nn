import pandas as pd
import numpy as np
import random as rg

rg = np.random.default_rng()


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
