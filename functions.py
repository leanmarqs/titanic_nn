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
    weights = rg.random((1, len(values)))[0]
    data_final = pd.DataFrame(features,
                              columns=['Pclass', 'Sex', 'Age', 'Siblings/Spouses Aboard', 'Parents/Children Aboard',
                                       'Fare'])
    data_final['values'] = values
    data_final['targets'] = targets
    return data_final, weights


def save_data(data, output_file):
    with open(output_file, 'w') as f:
        f.write(data.to_string())
