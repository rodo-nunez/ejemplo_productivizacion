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
parser.add_argument('--modo_prueba', default=f'{params.bool_modo_prueba_por_defecto}', help='Nos dice si estamos ejecutando el código en modo de pruebas o no. Un `True` hace que el código se ejecute mucho más rápido y ejecute lo escencial para verificar que se ejecuta sin bugs que rompen el código. Si es `False`, el código se ejecuta entero, lo que debería demorar mucho más.')

try:
    args = parser.parse_args()
except argparse.ArgumentTypeError as e:
    print(f"Invalid argument: {e}")

# Info ---------------------------------------- 

print(f"---------------------------------- \nComenzando proceso de entrenamiento de modelos \n----------------------------------")

# Preproceso ---------------------------------------- 

logging.info("Ejecutando a01")
os.system(f"python preprocessing/a01_preproceso_general.py --modo_prueba {args.modo_prueba}")
logging.info("Ejecutando a02")
os.system(f"python preprocessing/a02_feature_engineering.py --modo_prueba {args.modo_prueba}")
logging.info("Ejecutando a03")
os.system(f"python preprocessing/a03_division_train_test.py --modo_prueba {args.modo_prueba}")
logging.info("Ejecutando a04")
os.system(f"python preprocessing/a04_preproceso_post_division_train_test.py --modo_prueba {args.modo_prueba}")
logging.info("Ejecutando a05")
os.system(f"python preprocessing/a05_preproceso_dependiente_del_modelo.py --modo_prueba {args.modo_prueba}")

# Modelo ---------------------------------------- 

logging.info("Ejecutando b01")
os.system(f"python models/b01_entrenamiento_xgboost.py --modo_prueba {args.modo_prueba}")

logging.info("Terminado p01")