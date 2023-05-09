from functions import generate_data, prepare_data, save_data

data, weights = prepare_data('dataset/titanic_original.csv')
save_data(data, 'dataset/titanic_data.txt')
