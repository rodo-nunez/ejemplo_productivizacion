import pandas as pd
pd.options.display.max_columns = None
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

entrenamiento_sufix = params.get_entrenamiento_sufix(eval(args.bool_entrtenamiento))

# Lectura del modelo ---------------------------------------- 

model = joblib.load('files/modeling_output/model_fit/b01_best_xgb_model.joblib')

# Lectura de datos ---------------------------------------- 

feature_test_selected = joblib.load(f'files/datasets/intermediate/a05_feature_test_selected{entrenamiento_sufix}.pkl')
data_features = pd.read_feather(f"files/datasets/intermediate/a03_test_features{entrenamiento_sufix}.feather")
data_ids = pd.read_feather(f"files/datasets/intermediate/a03_test_ids{entrenamiento_sufix}.feather")

data = pd.concat([data_ids, data_features], axis = 1)

# Evaluar datos nuevos con modelo ---------------------------------------- 

y_pred = pd.DataFrame(model.predict(feature_test_selected), columns=['y_pred']).reset_index(drop=True)
y_pred_proba = pd.DataFrame(model.predict_proba(feature_test_selected)[:, 1], columns=['y_pred_proba']).reset_index(drop=True)

# Unimos respuesta con IDs ---------------------------------------- 

ids_y_respuesta_modelo = pd.concat([data_ids.reset_index(drop=True), y_pred, y_pred_proba], axis = 1)
data_y_respuesta_modelo = pd.concat([data.reset_index(drop=True), y_pred, y_pred_proba], axis = 1)

# Guardar resultados ---------------------------------------- 

ids_y_respuesta_modelo.to_feather(f"files/datasets/output/c01_resultados_con_ids{entrenamiento_sufix}.feather")
data_y_respuesta_modelo.to_feather(f"files/datasets/output/c01_resultados_con_datos_enteros{entrenamiento_sufix}.feather")