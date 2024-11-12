# Librerias ---------------------------------------- 

import pandas as pd
pd.options.display.max_columns = None
import numpy as np

# Leer input ---------------------------------------- 

data = pd.read_feather("files/datasets/intermediate/a01_datos_preprocesados.feather")

# Ingenieria de caractaristicas ----------------------------------------

data['internet_multilines'] = np.where(~data['internet_service'].isna() & ~data['multiple_lines'].isna(), ' Ambos',
                                       np.where(data['internet_service'].isna() & ~data['multiple_lines'].isna(), 'Sólo Teléfono',
                                                np.where(~data['internet_service'].isna() & data['multiple_lines'].isna(), 'Sólo Internet',
                                                         'no_inf')))

data['automatic_pay'] = np.where(data['payment_method'].isin(['Bank transfer (automatic)', 'Credit card (automatic)']), 
                                 'automatic', 
                                 'manual')

data['duration_days'] = (data['end_date'] - data['begin_date']).dt.days

data['extra_payment'] = data['total_charges'] - data['monthly_charges'] * data['duration_months']

# Rellenamos valores ausentes en las columnas donde aún existen los mismos
# Usamos 'No' asumiendo que los mismos no contrataron o aún no contratan tal servicio
col_nan_values = ['online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies']
for col in col_nan_values:
    if data[col].dtype.name == 'category':
        if 'No' not in data[col].cat.categories:
            data[col] = data[col].cat.add_categories('No')
        data[col] = data[col].fillna('No')

# Verificar columna 'multiple_lines' si es categórica
if data['multiple_lines'].dtype.name == 'category':
    data['multiple_lines'] = data['multiple_lines'].cat.rename_categories({'No': '0', 'Yes': '1'})
else:
    # caso de no ser categorico
    data['multiple_lines'] = data['multiple_lines'].replace({'No': '0', 'Yes': '1'})


for row in range(len(data)):
    
    if pd.isna(data.loc[row, 'multiple_lines']):
        data.loc[row, 'atleast_one_line'] = 0
    
    else:
        data.loc[row, 'atleast_one_line'] = 1
        
data['atleast_one_line'] = data['atleast_one_line'].astype('int')

for col in data.select_dtypes(['category']).columns:
    if -1 not in data[col].cat.categories:
        data[col] = data[col].cat.add_categories([-1])

# rellenar los valores nulos con -1
data = data.fillna(-1)

data['target'] = data.pop('target')

# TODO: Mover esta funcion de aqui y de a01 a la carpeta functions
def categorical_value(df, columns):
    for column in columns:
        df[column] = df[column].astype('category')
    return None

# convertimos a categorico la columna faltante
categorical_value(data, ['internet_multilines', 'automatic_pay', 'senior_citizen'])

# Escritura de output ---------------------------------------- 

data.to_feather("files/datasets/intermediate/a02_feature_engineering_done.feather")