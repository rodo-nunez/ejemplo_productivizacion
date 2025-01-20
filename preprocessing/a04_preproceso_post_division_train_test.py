# Librerias ---------------------------------------- 

import pandas as pd
pd.options.display.max_columns = None
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pickle
import joblib

import argparse
import sys, os
sys.path.append(os.getcwd()) # Esto es para agregar al path la ruta de ejecución actual y poder importar respecto a la ruta del proyecto, desde donde se debe ejecutar el código
import params as params

# Argumentos por linea de comandos ---------------------------------------- 

parser = argparse.ArgumentParser()
parser.add_argument('--modo_prueba', default=f'{params.bool_modo_prueba_por_defecto}', help='Nos dice si estamos ejecutando el código en modo de pruebas o no. Un `True` hace que el código se ejecute mucho más rápido y ejecute lo escencial para verificar que se ejecuta sin bugs que rompen el código. Si es `False`, el código se ejecuta entero, lo que debería demorar mucho más.')
parser.add_argument('--bool_entrtenamiento', default=f'{params.bool_entrtenamiento_por_defecto}', help='Nos dice si estamos ejecutando el código en para entrenar el modelo o no. Si no lo hacemos para entrenar el modelo, es porque lo usamos para usar el modelo ya entrenado y evaluar datos nuevos.')
parser.add_argument('--periodo', default=f'{params.periodo_YYYYMM_por_defecto}', help='Año y mes con el que evaluaremos el modelo.')

try:
    args = parser.parse_args()
except argparse.ArgumentTypeError as e:
    print(f"Invalid argument: {e}")

# Leer input ---------------------------------------- 

train_features = pd.read_feather("files/datasets/intermediate/a03_train_features.feather")
train_target = pd.read_csv("files/datasets/intermediate/a03_train_target.csv")
valid_features = pd.read_feather("files/datasets/intermediate/a03_valid_features.feather")
valid_target = pd.read_csv("files/datasets/intermediate/a03_valid_target.csv")
test_features = pd.read_feather("files/datasets/intermediate/a03_test_features.feather")
test_target = pd.read_csv("files/datasets/intermediate/a03_test_target.csv")

# Escalamiento y OHE ---------------------------------------- 

cat_cols = train_features.select_dtypes(include=['category', 'object']).columns.tolist()
num_cols = train_features.select_dtypes(include=['float64', 'int64']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop=None, sparse_output=False), cat_cols)
    ],
    force_int_remainder_cols=False,
    remainder='passthrough'
)

feature_train_transformed = preprocessor.fit_transform(train_features) #! Creamos la lógica del escalador con el dataset de train. Luego lo aplicamos a los otros
feature_valid_transformed = preprocessor.transform(valid_features)
feature_test_transformed = preprocessor.transform(test_features)

feature_train_transformed_df = pd.DataFrame(
    feature_train_transformed, 
    columns=preprocessor.get_feature_names_out(), 
    index=train_features.index 
)
feature_valid_transformed_df = pd.DataFrame(
    feature_valid_transformed, 
    columns=preprocessor.get_feature_names_out(), 
    index=valid_features.index 
)
feature_test_transformed_df = pd.DataFrame(
    feature_test_transformed, 
    columns=preprocessor.get_feature_names_out(), 
    index=test_features.index 
)

# Escribir outputs ---------------------------------------- 

with open('files/datasets/intermediate/a04_feature_train_transformed_df.pkl', 'wb') as file: 
    pickle.dump(feature_train_transformed_df, file) 
with open('files/datasets/intermediate/a04_feature_valid_transformed_df.pkl', 'wb') as file: 
    pickle.dump(feature_valid_transformed_df, file) 
with open('files/datasets/intermediate/a04_feature_test_transformed_df.pkl', 'wb') as file: 
    pickle.dump(feature_test_transformed_df, file) 
joblib.dump(preprocessor, 'files/datasets/intermediate/a04_preprocessor.pkl')
