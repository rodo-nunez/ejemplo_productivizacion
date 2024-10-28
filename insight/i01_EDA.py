# Librerias ---------------------------------------- 

import pandas as pd

# Lectura de datos ---------------------------------------- 

df_contract = pd.read_csv('datasets/contract.csv')
df_personal = pd.read_csv('datasets/personal.csv')
df_internet = pd.read_csv('datasets/internet.csv')
df_phone = pd.read_csv('datasets/phone.csv')

# Info basica ---------------------------------------- 

def data_info(df, nombre_df):
    print(f'Datos {nombre_df}:')
    df.head()

    print(f'Información de {nombre_df}:')
    df.info()

data_info(df_contract, 'df_contract')
data_info(df_personal, 'df_personal')
data_info(df_internet, 'df_internet')
data_info(df_phone, 'df_phone')

# Valores únicos y la frecuencia ---------------------------------------- 

def unique_values(df, df_name):
    for column in df.columns:
        print(f"Valores únicos en '{column}' de {df_name}: {df[column].unique()}")

def values_freq(df, df_name):
    for column in df.columns:
        print(f"Conteo de valores en '{column}' de {df_name}:")
        print(df[column].value_counts(dropna=False))

unique_values(df_contract, 'df_contract')
values_freq(df_contract, 'df_contract')
unique_values(df_personal, 'df_personal')
values_freq(df_personal, 'df_personal')
unique_values(df_internet, 'df_internet')
values_freq(df_internet, 'df_internet')
unique_values(df_phone, 'df_phone')
values_freq(df_phone, 'df_phone')
