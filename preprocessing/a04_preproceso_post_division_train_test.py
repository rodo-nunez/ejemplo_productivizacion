# Librerias ---------------------------------------- 

import pandas as pd
pd.options.display.max_columns = None
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pickle
import joblib

# Leer input ---------------------------------------- 

train_features = pd.read_feather("files/datasets/intermediate/a03_train_features.feather")
train_target = pd.read_csv("files/datasets/intermediate/a03_train_target.csv")
valid_features = pd.read_feather("files/datasets/intermediate/a03_valid_features.feather")
valid_target = pd.read_csv("files/datasets/intermediate/a03_valid_target.csv")
test_features = pd.read_feather("files/datasets/intermediate/a03_test_features.feather")
test_target = pd.read_csv("files/datasets/intermediate/a03_test_target.csv")

# Escalamiento ---------------------------------------- 

cat_cols = train_features.select_dtypes(include=['category']).columns.tolist()
num_cols = train_features.select_dtypes(include=['float64', 'int64']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop=None, sparse_output=False), cat_cols)
    ],
    remainder='passthrough'
)

feature_train_transformed = preprocessor.fit_transform(train_features) #! Creamos la lógica del escalador con el dataset de train. Luego lo aplicamos a los otros
feature_valid_transformed = preprocessor.transform(valid_features)
feature_test_transformed = preprocessor.transform(test_features)

# Escribir outputs ---------------------------------------- 

with open('files/datasets/intermediate/a04_feature_train_transformed.pkl', 'wb') as file: 
    pickle.dump(feature_train_transformed, file) 
with open('files/datasets/intermediate/a04_feature_valid_transformed.pkl', 'wb') as file: 
    pickle.dump(feature_valid_transformed, file) 
with open('files/datasets/intermediate/a04_feature_test_transformed.pkl', 'wb') as file: 
    pickle.dump(feature_test_transformed, file) 
joblib.dump(preprocessor, 'files/datasets/intermediate/a04_preprocessor.pkl')
