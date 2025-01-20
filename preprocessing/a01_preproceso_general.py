# Librerias ----------------------------------------

import numpy as np
import argparse
import matplotlib.pyplot as plt

import pandas as pd
pd.options.display.max_columns = None
import joblib

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, auc

from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from boruta import BorutaPy

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
    
# Lectura de datos ----------------------------------------

df_contract = pd.read_csv('files/datasets/input/contract.csv')
df_personal = pd.read_csv('files/datasets/input/personal.csv')
df_internet = pd.read_csv('files/datasets/input/internet.csv')
df_phone = pd.read_csv('files/datasets/input/phone.csv')

if (args.modo_prueba == "True") | (args.modo_prueba == True):
    df_contract = df_contract.head(params.n_filas_en_modo_prueba)
    df_personal = df_personal.head(params.n_filas_en_modo_prueba)
    df_internet = df_internet.head(params.n_filas_en_modo_prueba)
    df_phone = df_phone.head(params.n_filas_en_modo_prueba)

# Reemplazo de nulos que no introducen data leakage ----------------------------------------
# ! Estos remplazos no introducen data leakage. Si queremos agregar otros reemplazos hay que analizar si es bueno ponerlos acá o no.

def replace_spaces(DataFrames):
    for df in DataFrames:
        df.replace(' ', np.nan, inplace=True)

dataframes = [df_contract, df_personal, df_internet, df_phone]
replace_spaces(dataframes)

df_contract['TotalCharges'] = df_contract['TotalCharges'].fillna(0)

# Duplicados ----------------------------------------

def duplicates(dfs):
    duplicated = {}
    for i, df in enumerate(dfs):
        print(i)
        count = df.duplicated().sum()
        duplicated[f'DataFrame_{i+1}'] = count
    return duplicated


dfs = [df_contract, df_personal, df_internet, df_phone]
duplicate = duplicates(dfs)

# Tipos de datos ----------------------------------------

# df_contract
df_contract['BeginDate'] = pd.to_datetime(
    df_contract['BeginDate'], errors='coerce')
# Fecha de referencia para contratos sin EndDate
df_contract['EndDate'] = pd.to_datetime(
    df_contract['EndDate'], errors='coerce')
reference_date = pd.to_datetime('2020-02-01') # TODO Pasar a archivo de variables globales


# Función que calcula duración en meses
def calculate_duration_months(begin_date, end_date):
    if pd.isna(begin_date):
        return np.nan
    if pd.isna(end_date):
        end_date = reference_date
    duration = (end_date.year - begin_date.year) * \
        12 + end_date.month - begin_date.month
    return duration


# Duración en meses
df_contract['BeginDate'] = pd.to_datetime(
    df_contract['BeginDate'], errors='coerce')
df_contract['duration_months'] = df_contract.apply(
    lambda row: calculate_duration_months(row['BeginDate'], row['EndDate']),
    axis=1
)

# Columna target
df_contract['target'] = (df_contract['EndDate'] < reference_date).astype(int)

# Convetir a categorico

def categorical_value(df, columns):
    for column in columns:
        df[column] = df[column].astype('category')
    return None

categorical_contract = ['Type', 'PaymentMethod', 'PaperlessBilling']
categorical_value(df_contract, categorical_contract)

# Convirtiendo TotalCharges a númerico
df_contract['TotalCharges'] = pd.to_numeric(
    df_contract['TotalCharges'], errors='coerce')

# Reemplazamos NaN con el máximo hallado
df_contract['EndDate'] = df_contract['EndDate'].fillna(
    df_contract['EndDate'].max())

# df_personal
categorical_personal = ['gender', 'Partner', 'Dependents']
categorical_value(df_personal, categorical_personal)

# df_internet
categorical_internet = ['InternetService', 'OnlineSecurity', 'OnlineBackup',
                        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
categorical_value(df_internet, categorical_internet)

# df_phone
categorical_value(df_phone, ['MultipleLines'])

# Combinando DataFrames ----------------------------------------

data = df_contract.merge(df_personal, how='left', on='customerID')
data = data.merge(df_internet, how='left', on='customerID')
data = data.merge(df_phone, how='left', on='customerID')

# Nombres de columnas ----------------------------------------

for i in data.columns:
    data.columns = data.columns.str.lower()

data = data.rename(columns={'customerid': 'customer_id', 'begindate': 'begin_date',
                            'enddate': 'end_date', 'paperlessbilling': 'paperless_billing',
                            'paymentmethod': 'payment_method', 'monthlycharges': 'monthly_charges',
                            'totalcharges': 'total_charges', 'internetservice': 'internet_service',
                            'onlinesecurity': 'online_security', 'onlinebackup': 'online_backup',
                            'deviceprotection': 'device_protection', 'techsupport': 'tech_support',
                                                'streamingtv': 'streaming_tv', 'streamingmovies': 'streaming_movies',
                                                'multiplelines': 'multiple_lines', 'seniorcitizen': 'senior_citizen',
                                                'type': 'contract_type'})

# Ajuste de tipos de datos ----------------------------------------

# columnas categóricas y numéricas
cat_cols = data.select_dtypes(include=['category']).columns.tolist()
num_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Categóricas a tipo string
data[cat_cols] = data[cat_cols].astype(str)

# Guardamos datos ----------------------------------------

data.to_feather("files/datasets/intermediate/a01_datos_preprocesados.feather")
joblib.dump(cat_cols, 'files/datasets/intermediate/a01_cat_cols.pkl')
joblib.dump(num_cols, 'files/datasets/intermediate/a01_num_cols.pkl')
