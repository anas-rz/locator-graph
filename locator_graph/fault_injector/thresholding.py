import pickle ,json

def load_dict(filename):

    if filename.endswith('.pkl'):
        with open(filename, 'rb') as file:
            return pickle.load(file)
    elif filename.endswith('.json'):
        with open(filename, 'r') as file:
            return json.load(file)
    else:
        raise ValueError("Filename must end in .pkl or .json")

def read_txt_to_list(filename):

    try:
        with open(filename, 'r') as file:  # Open file in read mode ('r')
            lines = [line.strip() for line in file]  # Read and strip newlines
        return lines
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return []  # Return an empty list if file not found



import numpy as np
from scipy.stats import ttest_ind

def cohens_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)


def is_killed(base_model, test_model, test_type, dicts):
    accs_base = []
    accs_test = []
    for _d in dicts:
        try:
            accs_base.append(_d[base_model][test_type][0])
        except KeyError:
            pass  # Skip if the key doesn't exist

        try:
            accs_test.append(_d[test_model][test_type][0])
        except KeyError:
            pass  # Skip if the key doesn't exist
    t_statistic, p_value = ttest_ind(accs_base, accs_test)
    effect_size = cohens_d(accs_base, accs_test)
    # print(f"T-statistic: {t_statistic:.3f}")
    # print(f"P-value: {p_value:.3f}")
    # print(f"Cohen's d: {effect_size:.3f}")
    is_killed = effect_size >= 0.001 and p_value < 0.05
    # print(f"Mutation killed: {is_killed}")
    return is_killed


paths = read_txt_to_list('all_records.txt')
dicts = [load_dict(_p) for _p in paths]


# test_type = 'accuracy'
# killed_train = dict()
# for _m in test_models:
#     killed_check = is_killed(base_model, _m, test_type, dicts)
#     killed_train[_m] = killed_check

# test_type = 'val_accuracy'
# killed_val = dict()
# for _m in test_models:
#     killed_check = is_killed(base_model, _m, test_type, dicts)
#     killed_val[_m] = killed_check


test_type = 'loss'
killed_train_loss = dict()
for _m in test_models:
    killed_check = is_killed(base_model, _m, test_type, dicts)
    killed_train_loss[_m] = killed_check

test_type = 'val_loss'
killed_val_loss = dict()
for _m in test_models:
    killed_check = is_killed(base_model, _m, test_type, dicts)
    killed_val_loss[_m] = killed_check


import pandas as pd

# Create DataFrame using keys from dictionaries
df = pd.DataFrame({'killed_on_train': killed_train, 'killed_on_val': killed_val})
df.index.name = 'model_name'  # Set the index name

# Display
print(df.head().to_markdown(index=True, numalign='left', stralign='left'))
