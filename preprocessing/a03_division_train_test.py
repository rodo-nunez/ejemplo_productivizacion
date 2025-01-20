# Librerias ---------------------------------------- 

import pandas as pd
pd.options.display.max_columns = None
from sklearn.model_selection import train_test_split
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

data = pd.read_feather("files/datasets/intermediate/a02_feature_engineering_done.feather")

# División entre train y test ---------------------------------------- 

if (args.modo_prueba == "True") | (args.modo_prueba == True):
    reference_date = data['begin_date'].median()
else:
    reference_date = pd.to_datetime('2020-02-01') # TODO Pasar a archivo de variables globales

valid_set = data[data['begin_date'] >= reference_date]
train_test_set = data[data['begin_date'] < reference_date]

train_set, test_set = train_test_split(train_test_set, test_size=0.25, random_state=12345, stratify=train_test_set['target'])

# Division entre features y target ---------------------------------------- 

id_columns = ['customer_id', 'begin_date', 'end_date']

train_features = train_set.drop(columns=id_columns + ['target'])
train_target = train_set['target']
train_ids = train_set[id_columns]

valid_features = valid_set.drop(columns=id_columns + ['target'])
valid_target = valid_set['target']
valid_ids = valid_set[id_columns]

test_features = test_set.drop(columns=id_columns + ['target'])
test_target = test_set['target']
test_ids = test_set[id_columns]

# Escribir outputs ---------------------------------------- 

train_features.to_feather("files/datasets/intermediate/a03_train_features.feather")
train_target.to_csv("files/datasets/intermediate/a03_train_target.csv", index=False)
train_ids.to_feather("files/datasets/intermediate/a03_train_ids.feather")

valid_features.to_feather("files/datasets/intermediate/a03_valid_features.feather")
valid_target.to_csv("files/datasets/intermediate/a03_valid_target.csv", index=False)
valid_ids.to_feather("files/datasets/intermediate/a03_valid_ids.feather")

test_features.to_feather("files/datasets/intermediate/a03_test_features.feather")
test_target.to_csv("files/datasets/intermediate/a03_test_target.csv", index=False)
test_ids.to_feather("files/datasets/intermediate/a03_test_ids.feather")
