# Librerias ---------------------------------------- 

import platform 
# from functions.crear_formatos_fecha import *

# Systema operativo ---------------------------------------- 

sistema_operativo = platform.system()

# Modo de prueba ---------------------------------------- 

bool_modo_prueba_por_defecto = False
n_filas_en_modo_prueba = 1000

# Entrenamiento o ejecucion ---------------------------------------- 

bool_entrtenamiento_por_defecto = True
def get_entrenamiento_sufix(bool_entrtenamiento):
    if bool_entrtenamiento:
        entrenamiento_sufix = ""
    else:
        entrenamiento_sufix = "_evaluacion"
    return entrenamiento_sufix

# Fecha por defecto ---------------------------------------- 

periodo_YYYYMM_por_defecto = "202311"

