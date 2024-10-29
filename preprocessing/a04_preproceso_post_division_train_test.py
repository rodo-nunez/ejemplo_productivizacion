# Librerias ---------------------------------------- 

import pandas as pd
pd.options.display.max_columns = None
import numpy as np

# Leer input ---------------------------------------- 

train_features = pd.read_feather("files/datasets/intermediate/a03_train_features.feather")
train_target = pd.read_csv("files/datasets/intermediate/a03_train_target.csv")
valid_features = pd.read_feather("files/datasets/intermediate/a03_valid_features.feather")
valid_target = pd.read_csv("files/datasets/intermediate/a03_valid_target.csv")
test_features = pd.read_feather("files/datasets/intermediate/a03_test_features.feather")
test_target = pd.read_csv("files/datasets/intermediate/a03_test_target.csv")

# Escalamiento ---------------------------------------- 

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols)
    ],
    remainder='passthrough'
)

feature_train_transformed = preprocessor.fit_transform(train_features)
feature_valid_transformed = preprocessor.transform(valid_features)
feature_test_transformed = preprocessor.transform(test_features)