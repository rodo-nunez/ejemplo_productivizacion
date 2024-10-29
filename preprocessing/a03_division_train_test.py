# Librerias ---------------------------------------- 

import pandas as pd
pd.options.display.max_columns = None
from sklearn.model_selection import train_test_split

# Leer input ---------------------------------------- 

data = pd.read_feather("files/datasets/intermediate/a02_feature_engineering_done.feather")

# División entre train y test ---------------------------------------- 

reference_date = pd.to_datetime('2020-02-01') # TODO Pasar a archivo de variables globales

valid_set = data[data['begin_date'] >= reference_date]
train_test_set = data[data['begin_date'] < reference_date]

train_set, test_set = train_test_split(train_test_set, test_size=0.25, random_state=12345, stratify=train_test_set['target'])

# Division entre features y target ---------------------------------------- 

columns_to_drop = ['customer_id', 'begin_date', 'end_date']

train_features = train_set.drop(columns=columns_to_drop + ['target'])
train_target = train_set['target']

valid_features = valid_set.drop(columns=columns_to_drop + ['target'])
valid_target = valid_set['target']

test_features = test_set.drop(columns=columns_to_drop + ['target'])
test_target = test_set['target']

# Escribir outputs ---------------------------------------- 

train_features.to_feather("files/datasets/intermediate/a03_train_features.feather")
train_target.to_csv("files/datasets/intermediate/a03_train_target.csv")
valid_features.to_feather("files/datasets/intermediate/a03_valid_features.feather")
valid_target.to_csv("files/datasets/intermediate/a03_valid_target.csv")
test_features.to_feather("files/datasets/intermediate/a03_test_features.feather")
test_target.to_csv("files/datasets/intermediate/a03_test_target.csv")



