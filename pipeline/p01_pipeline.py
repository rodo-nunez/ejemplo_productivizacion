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
parser.add_argument('--bool_entrtenamiento', default=f'{params.bool_entrtenamiento_por_defecto}', help='Nos dice si estamos ejecutando el código en para entrenar el modelo o no. Si no lo hacemos para entrenar el modelo, es porque lo usamos para usar el modelo ya entrenado y evaluar datos nuevos.')
parser.add_argument('--periodo', default=f'{params.periodo_YYYYMM_por_defecto}', help='Año y mes con el que evaluaremos el modelo.')

try:
    args = parser.parse_args()
except argparse.ArgumentTypeError as e:
    print(f"Invalid argument: {e}")

# Info ---------------------------------------- 

print(f"---------------------------------- \nComenzando proceso de entrenamiento de modelos \n----------------------------------")

# Preproceso ---------------------------------------- 

logging.info("Ejecutando a01")
os.system(f"python preprocessing/a01_preproceso_general.py --modo_prueba {args.modo_prueba} --bool_entrtenamiento {args.bool_entrtenamiento} --periodo {args.periodo}")
logging.info("Ejecutando a02")
os.system(f"python preprocessing/a02_feature_engineering.py --modo_prueba {args.modo_prueba} --bool_entrtenamiento {args.bool_entrtenamiento} --periodo {args.periodo}")
logging.info("Ejecutando a03")
os.system(f"python preprocessing/a03_division_train_test.py --modo_prueba {args.modo_prueba} --bool_entrtenamiento {args.bool_entrtenamiento} --periodo {args.periodo}") # TODO Este script podría no ser necesario en la evaluación del modelo, al igual que varios otros. Para eso puede servir el booleanos "args.bool_entrtenamiento". Para ejecutarlos dentro de un "if". También es importante que los outputs se evaluación no sobre escriban a los de entrenamiento, ni viceversa.
logging.info("Ejecutando a04")
os.system(f"python preprocessing/a04_preproceso_post_division_train_test.py --modo_prueba {args.modo_prueba} --bool_entrtenamiento {args.bool_entrtenamiento} --periodo {args.periodo}")
logging.info("Ejecutando a05")
os.system(f"python preprocessing/a05_preproceso_dependiente_del_modelo.py --modo_prueba {args.modo_prueba} --bool_entrtenamiento {args.bool_entrtenamiento} --periodo {args.periodo}")

# Modelo ---------------------------------------- 

logging.info("Ejecutando b01")
os.system(f"python models/b01_entrenamiento_xgboost.py --modo_prueba {args.modo_prueba} --periodo {args.periodo}")

# Evaluacion ---------------------------------------- 

logging.info("Ejecutando c01")
os.system(f"python execution/c01_ejecusion_modelo.py --modo_prueba {args.modo_prueba} --bool_entrtenamiento {args.bool_entrtenamiento} --periodo {args.periodo}")

logging.info("Terminado p01")