# Librerias ---------------------------------------- 

import pandas as pd
pd.options.display.max_columns = None
import pickle
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib

# Leer input ---------------------------------------- 

with open('files/datasets/intermediate/a04_feature_train_transformed.pkl', 'rb') as file: 
    feature_train_transformed = pickle.load(file) 
with open('files/datasets/intermediate/a04_feature_valid_transformed.pkl', 'rb') as file: 
    feature_valid_transformed = pickle.load(file) 
with open('files/datasets/intermediate/a04_feature_test_transformed.pkl', 'rb') as file: 
    feature_test_transformed = pickle.load(file) 

train_target = pd.read_csv("files/datasets/intermediate/a03_train_target.csv")
valid_target = pd.read_csv("files/datasets/intermediate/a03_valid_target.csv")
test_target = pd.read_csv("files/datasets/intermediate/a03_test_target.csv")

# TODO Arreglar este script. Las dimensiones no coinsiden

preprocessor = joblib.load('files/datasets/intermediate/a04_preprocessor.pkl')
# cat_cols = joblib.load('files/datasets/intermediate/a01_cat_cols.pkl')
# num_cols = joblib.load('files/datasets/intermediate/a01_num_cols.pkl')

len(feature_train_transformed[0])

# OHE ---------------------------------------- 

# train_features = pd.read_feather("files/datasets/intermediate/a03_train_features.feather").head()
# train_features = pd.read_feather("files/datasets/intermediate/a01_datos_preprocesados.feather").head()
cat_cols = train_features.select_dtypes(include=['category']).columns.tolist() # TODO Obtener esta variable una única vez y guardar el resultado
num_cols = train_features.select_dtypes(include=['float64', 'int64']).columns.tolist()

#Conservando nombres originales y creando DF
ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)

new_feature_names = num_cols + list(ohe_feature_names)

#Convetir a DataFrame
features_train_OHE = pd.DataFrame(feature_train_transformed, columns=new_feature_names)
features_valid_OHE = pd.DataFrame(feature_valid_transformed, columns=new_feature_names)
features_test_OHE = pd.DataFrame(feature_test_transformed, columns=new_feature_names)

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

# Escribir outputs ---------------------------------------- 

joblib.dump(feature_train_selected, 'files/datasets/intermediate/a05_feature_train_selected.pkl')
joblib.dump(feature_valid_selected, 'files/datasets/intermediate/a05_feature_valid_selected.pkl')
joblib.dump(feature_test_selected, 'files/datasets/intermediate/a05_feature_test_selected.pkl')
joblib.dump(feature_train_balanced, 'files/datasets/intermediate/a05_feature_train_balanced.pkl')
joblib.dump(target_train_balanced, 'files/datasets/intermediate/a05_target_train_balanced.pkl')

