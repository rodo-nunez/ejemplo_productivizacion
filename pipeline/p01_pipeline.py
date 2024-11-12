# Librerias ----------------------------------------

import os, sys
import argparse
sys.path.append(os.getcwd()) # Esto es para agregar al path la ruta de ejecución actual y poder importar respecto a la ruta del proyecto, desde donde se debe ejecutar el código
import params as params
import logging

# Configuracion del archivo de log ---------------------------------------- 
logging.basicConfig(filename='files/modeling_output/logs/p01_pipeline_entrenamiento.log', level=logging.INFO, format='%(asctime)s - %(message)s', filemode='w')

# Argumentos por linea de comandos ---------------------------------------- 

parser = argparse.ArgumentParser()
parser.add_argument('--periodo', default=f'{params.periodo_YYYYMM_por_defecto}', help='periodo en formato YYYYMM')

try:
    args = parser.parse_args()
except argparse.ArgumentTypeError as e:
    print(f"Invalid argument: {e}")
    
# Definir extension de ejecutables ---------------------------------------- 

if params.sistema_operativo == 'Windows':
        extension_binarios = ".exe"
else:
        extension_binarios = ""

# Info ---------------------------------------- 

print(f"---------------------------------- \nComenzando proceso de entrenamiento de modelos \n----------------------------------")

# Preproceso ---------------------------------------- 

logging.info("Ejecutando a01")
os.system(f"python{extension_binarios} preprocessing/a01_preproceso_general.py")
logging.info("Ejecutando a02")
os.system(f"python{extension_binarios} preprocessing/a02_feature_engineering.py")
logging.info("Ejecutando a03")
os.system(f"python{extension_binarios} preprocessing/a03_division_train_test.py")
logging.info("Ejecutando a04")
os.system(f"python{extension_binarios} preprocessing/a04_preproceso_post_division_train_test.py")
logging.info("Ejecutando a05")
os.system(f"python{extension_binarios} preprocessing/a05_preproceso_dependiente_del_modelo.py")
logging.info("Ejecutando a06")
os.system(f"python{extension_binarios} preprocessing/a06_pipeline_y_bootstraping_para_comparar_modelos.py")

# Modelo ---------------------------------------- 

logging.info("Ejecutando b01")
os.system(f"python{extension_binarios} models/b01_entrenamiento_xgboost.py")

logging.info("Terminado p01")