# Librerias ----------------------------------------

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
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

# Lectura de datos ----------------------------------------

df_contract = pd.read_csv('files/datasets/input/contract.csv')
df_personal = pd.read_csv('files/datasets/input/personal.csv')
df_internet = pd.read_csv('files/datasets/input/internet.csv')
df_phone = pd.read_csv('files/datasets/input/phone.csv')

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
reference_date = pd.to_datetime('2020-02-01')


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
    return df


categorical_contract = ['Type', 'PaymentMethod', 'PaperlessBilling']
df_contract = categorical_value(df_contract, categorical_contract)

# Convirtiendo TotalCharges a númerico
df_contract['TotalCharges'] = pd.to_numeric(
    df_contract['TotalCharges'], errors='coerce')

# Reemplazamos NaN con el máximo hallado
df_contract['EndDate'] = df_contract['EndDate'].fillna(
    df_contract['EndDate'].max())

# df_personal
categorical_personal = ['gender', 'Partner', 'Dependents']
df_personal = categorical_value(df_personal, categorical_personal)

# df_internet
categorical_internet = ['InternetService', 'OnlineSecurity', 'OnlineBackup',
                        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
df_internet = categorical_value(df_internet, categorical_internet)

# df_phone
df_phone = categorical_value(df_phone, ['MultipleLines'])

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
