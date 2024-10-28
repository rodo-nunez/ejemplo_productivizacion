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

df_contract = pd.read_csv('datasets/contract.csv')
df_personal = pd.read_csv('datasets/personal.csv')
df_internet = pd.read_csv('datasets/internet.csv')
df_phone = pd.read_csv('datasets/phone.csv')

# Reemplazo de nulos ----------------------------------------

# TODO: Hacer esto después de dividir entre train y test para evitar Data Leakage


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

# convertimos a categorico la columna faltante
categorical_value(data, ['internet_multilines', 'automatic_pay', 'senior_citizen'])

# División entre train y test ---------------------------------------- 

valid_set = data[data['begin_date'] >= reference_date]
train_test_set = data[data['begin_date'] < reference_date]

train_set, test_set = train_test_split(train_test_set, test_size=0.25, random_state=12345, stratify=train_test_set['target'])

columns_to_drop = ['customer_id', 'begin_date', 'end_date']

train_features = train_set.drop(columns=columns_to_drop + ['target'])
train_target = train_set['target']

valid_features = valid_set.drop(columns=columns_to_drop + ['target'])
valid_target = valid_set['target']

test_features = test_set.drop(columns=columns_to_drop + ['target'])
test_target = test_set['target']

# Ajuste de tipos de datos ---------------------------------------- 

# columnas categóricas y numéricas
cat_cols = train_features.select_dtypes(include=['category']).columns.tolist()
num_cols = train_features.select_dtypes(include=['float64', 'int64']).columns.tolist()

# TODO Hacerlo antes de dividir los datos
# Categóricas a tipo string
train_features[cat_cols] = train_features[cat_cols].astype(str)
valid_features[cat_cols] = valid_features[cat_cols].astype(str)
test_features[cat_cols] = test_features[cat_cols].astype(str)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols)
    ],
    remainder='passthrough'
)

feature_train_transformed = preprocessor.fit_transform(train_features)
feature_valid_transformed = preprocessor.transform(valid_features)
feature_test_transformed = preprocessor.transform(test_features)

# OHE ---------------------------------------- 

#Conservando nombres originales y creando DF
ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)

new_feature_names = num_cols + list(ohe_feature_names)

#Convetir a DataFrame
features_train_OHE = pd.DataFrame(feature_train_transformed, columns=new_feature_names)
features_valid_OHE = pd.DataFrame(feature_valid_transformed, columns=new_feature_names)
features_test_OHE =pd.DataFrame(feature_test_transformed, columns=new_feature_names)

# SMOTE ---------------------------------------- 

smote = SMOTE(random_state=12345)
feature_train_balanced, target_train_balanced = smote.fit_resample(features_train_OHE, train_target)

# Boruta feature selection ---------------------------------------- 

rf_boruta = RandomForestClassifier(n_jobs=-1, max_depth=5, random_state=12345)
boruta_selector = BorutaPy(rf_boruta, n_estimators='auto', perc=100, random_state=12345)

#Boruta al conjunto de datos balanceado
boruta_selector.fit(feature_train_balanced, target_train_balanced)

#Seleccionando las características relevantes
selected_features = features_train_OHE.columns[boruta_selector.support_].tolist()

feature_train_selected = feature_train_balanced[selected_features]
feature_valid_selected = features_valid_OHE[selected_features]
feature_test_selected = features_test_OHE[selected_features]

# CrossValidation ---------------------------------------- 

#Validación cruzada para verificar estabilidad de características seleccionadas
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=12345)

#Validando. cross_val_score para ver si las características seleccionadas son estables

cv_scores = cross_val_score(rf_boruta, feature_train_selected, target_train_balanced, cv=skf, scoring='accuracy')

# Pipelines ---------------------------------------- 

pipelines = {
    'RandomForest': Pipeline(steps=[
        ('classifier', RandomForestClassifier(random_state=12345))
    ]),
    'LogisticRegression': Pipeline(steps=[
        ('classifier', LogisticRegression(random_state=12345))
    ]),
    'MLPClassifier': Pipeline(steps=[
        ('classifier', MLPClassifier(random_state=12345))
    ]),
    'GradientBoosting': Pipeline(steps=[
        ('classifier', GradientBoostingClassifier(random_state=12345))
    ]),
    'AdaBoost': Pipeline(steps=[
        ('classifier', AdaBoostClassifier(random_state=12345))
    ]),
    'DecisionTree': Pipeline(steps=[
        ('classifier', DecisionTreeClassifier(random_state=12345))
    ]),
    'LGBMClassifier': Pipeline(steps=[
        ('classifier', LGBMClassifier(random_state=12345))
    ]),
    'XGBoost': Pipeline(steps=[
        ('classifier', XGBClassifier(random_state=12345))
    ])
}

# Bootstrap ---------------------------------------- 

for name, pipeline in pipelines.items():
    print(f"\nModelo: {name}")

    cv_scores = cross_val_score(pipeline, feature_train_selected, target_train_balanced, cv=skf, scoring='accuracy')
    print(f"Puntuaciones de validación cruzada: {cv_scores}")
    print(f"Promedio de las puntuaciones: {np.mean(cv_scores)}")
        
    pipeline.fit(feature_train_selected, target_train_balanced)
    valid_predictions = pipeline.predict(feature_valid_selected)
    valid_report = classification_report(valid_target, valid_predictions)
    print("Reporte de clasificación en conjunto de validación:")
    print(valid_report)
        
    test_predictions = pipeline.predict(feature_test_selected)
    test_report = classification_report(test_target, test_predictions)
    print("Reporte de clasificación en conjunto de prueba:")
    print(test_report)
        
    #Bootstraping
    n_iterations = 1000
    bootstrap_scores = []
    for i in range(n_iterations):
        indices = np.random.choice(range(len(feature_test_selected)), size=len(feature_test_selected), replace=True)
        feature_test_bootstrap = feature_test_selected.iloc[indices]
        target_test_bootstrap = test_target.iloc[indices]
        bootstrap_predictions = pipeline.predict(feature_test_bootstrap)
        bootstrap_scores.append(np.mean(bootstrap_predictions == target_test_bootstrap))
        
    print("Media de las puntuaciones Bootstrap:", np.mean(bootstrap_scores))
    print("Desviación estándar de las puntuaciones Bootstrap:", np.std(bootstrap_scores))

# GBC ---------------------------------------- 

#GB_model
gb_model = GradientBoostingClassifier()

#hallo hiperparametros
gb_param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0]
}

gb_grid_search = GridSearchCV(estimator=gb_model, param_grid=gb_param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
gb_grid_search.fit(feature_train_selected, target_train_balanced)

print("Mejor modelo Gradient Boosting:")
print(gb_grid_search.best_estimator_)

print("Mejores hiperparámetros:")
print(gb_grid_search.best_params_)

print("Mejor puntuación AUC-ROC:")
print(gb_grid_search.best_score_)

best_gb = gb_grid_search.best_estimator_
best_gb.fit(feature_train_selected, target_train_balanced)

y_pred = best_gb.predict(feature_test_selected)
y_pred_proba = best_gb.predict_proba(feature_test_selected)[:, 1]

print("Reporte de Clasificación:")
print(classification_report(test_target, y_pred))

print("Puntuación AUC-ROC:")
print(roc_auc_score(test_target, y_pred_proba))

fpr, tpr, _ = roc_curve(test_target, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Falsos Positivos')
plt.ylabel('Verdaderos positivos')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()

# MLPClassifier ---------------------------------------- 

#MLP_model
mlp_model = MLPClassifier(max_iter=1000)

mlp_param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
    'activation': ['tanh', 'relu'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001],
    'learning_rate': ['constant', 'adaptive']
}

mlp_grid_search = GridSearchCV(estimator=mlp_model, param_grid=mlp_param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)

mlp_grid_search.fit(feature_train_selected, target_train_balanced)

print("Mejor modelo MLP:")
print(mlp_grid_search.best_estimator_)

print("Mejores hiperparámetros:")
print(mlp_grid_search.best_params_)

print("Mejor puntuación AUC-ROC:")
print(mlp_grid_search.best_score_)

best_mlp = mlp_grid_search.best_estimator_
best_mlp.fit(feature_train_selected, target_train_balanced)

y_pred = best_mlp.predict(feature_test_selected)
y_pred_proba = best_mlp.predict_proba(feature_test_selected)[:, 1]

print("Reporte de Clasificación:")
print(classification_report(test_target, y_pred))

print("Puntuación AUC-ROC:")
print(roc_auc_score(test_target, y_pred_proba))

fpr, tpr, _ = roc_curve(test_target, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Falso Positivo')
plt.ylabel('Verdadero Positivo')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()

# RandomForestClassifier ---------------------------------------- 

#rf_model
rf_model = RandomForestClassifier()

rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

rf_grid_search = GridSearchCV(estimator=rf_model, param_grid=rf_param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)

rf_grid_search.fit(feature_train_selected, target_train_balanced)

print("Mejor modelo Random Forest:")
print(rf_grid_search.best_estimator_)

print("Mejores hiperparámetros:")
print(rf_grid_search.best_params_)

print("Mejor puntuación AUC-ROC:")
print(rf_grid_search.best_score_)

best_rf = rf_grid_search.best_estimator_
best_rf.fit(feature_train_selected, target_train_balanced)

y_pred = best_rf.predict(feature_test_selected)
y_pred_proba = best_rf.predict_proba(feature_test_selected)[:, 1]

print("Reporte de Clasificación:")
print(classification_report(test_target, y_pred))

print("Puntuación AUC-ROC:")
print(roc_auc_score(test_target, y_pred_proba))

fpr, tpr, _ = roc_curve(test_target, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Falso Positivo')
plt.ylabel('Verdadero Positivo')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()

# XGBClassifier ---------------------------------------- 

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

xgb_param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.7, 1.0]
}

xgb_grid_search = GridSearchCV(estimator=xgb_model, param_grid=xgb_param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)

xgb_grid_search.fit(feature_train_selected, target_train_balanced)

print("Mejor modelo XGBoost:")
print(xgb_grid_search.best_estimator_)

print("Mejores hiperparámetros:")
print(xgb_grid_search.best_params_)

print("Mejor puntuación AUC-ROC:")
print(xgb_grid_search.best_score_)

best_xgb = xgb_grid_search.best_estimator_
best_xgb.fit(feature_train_selected, target_train_balanced)

y_pred = best_xgb.predict(feature_test_selected)
y_pred_proba = best_xgb.predict_proba(feature_test_selected)[:, 1]

print("Reporte de Clasificación:")
print(classification_report(test_target, y_pred))

print("Puntuación AUC-ROC:")
print(roc_auc_score(test_target, y_pred_proba))

fpr, tpr, _ = roc_curve(test_target, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Falsos Positivos')
plt.ylabel('Verdaderos Positivos')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()


