# Librerias ---------------------------------------- 

import pandas as pd
pd.options.display.max_columns = None
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, auc
import joblib



import sys
if hasattr(sys, 'ps1') or 'IPython' in sys.modules:
    es_entorno_interactivo = True
else:
    es_entorno_interactivo = False 
    import matplotlib
    matplotlib.use('Agg') 

import logging
logging.basicConfig(filename='files/modeling_output/logs/b01_debug_entrenamiento_por_terminal.log', level=logging.INFO, format='%(asctime)s - %(message)s', filemode='w')


logging.info("---------- Leer datasets")


# Leer datasets ---------------------------------------- 

feature_train_selected = joblib.load('files/datasets/intermediate/a05_feature_train_selected.pkl')
feature_valid_selected = joblib.load('files/datasets/intermediate/a05_feature_valid_selected.pkl')
feature_test_selected = joblib.load('files/datasets/intermediate/a05_feature_test_selected.pkl')
feature_train_balanced = joblib.load('files/datasets/intermediate/a05_feature_train_balanced.pkl')
target_train_balanced = joblib.load('files/datasets/intermediate/a05_target_train_balanced.pkl')
test_target = pd.read_csv("files/datasets/intermediate/a03_test_target.csv")

logging.info("---------- XGBClassifier")
# XGBClassifier ---------------------------------------- 

xgb_model = XGBClassifier(eval_metric='logloss')

# xgb_param_grid = {
#     'n_estimators': [100, 200],
#     'learning_rate': [0.01, 0.1],
#     'max_depth': [3, 5, 7],
#     'subsample': [0.8, 1.0],
#     'colsample_bytree': [0.7, 1.0]
# }
xgb_param_grid = {
    'n_estimators': [5],
    'learning_rate': [0.1],
    'max_depth': [3],
    'subsample': [0.8],
    'colsample_bytree': [0.7]
}
logging.info("---------- GridSearch")

# GridSearch ---------------------------------------- 

xgb_grid_search = GridSearchCV(estimator=xgb_model, param_grid=xgb_param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)

xgb_grid_search.fit(feature_train_selected, target_train_balanced)

print("Mejor modelo XGBoost:")
print(xgb_grid_search.best_estimator_)

print("Mejores hiperparámetros:")
print(xgb_grid_search.best_params_)

print("Mejor puntuación AUC-ROC:")
print(xgb_grid_search.best_score_)

logging.info("---------- Eleccion del mejor XGB")

# Elección del mejor XGB ---------------------------------------- 

best_xgb = xgb_grid_search.best_estimator_
best_xgb.fit(feature_train_selected, target_train_balanced)

y_pred = best_xgb.predict(feature_test_selected)
y_pred_proba = best_xgb.predict_proba(feature_test_selected)[:, 1]

print("Reporte de Clasificación:")
print(classification_report(test_target, y_pred))

print("Puntuación AUC-ROC:")
print(roc_auc_score(test_target, y_pred_proba))

logging.info("---------- Grafico de ROC")


# Grafico de ROC ---------------------------------------- 

fpr, tpr, _ = roc_curve(test_target, y_pred_proba)
roc_auc = auc(fpr, tpr)

logging.info("---------- figure")

plt.figure()

logging.info("---------- plot")


plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

logging.info("---------- plt.xlim")

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Falsos Positivos')
plt.ylabel('Verdaderos Positivos')
plt.title('Curva ROC')
plt.legend(loc="lower right")

logging.info("---------- savefig")

plt.savefig('files/modeling_output/figures/b01_curva_roc_best_xgb.pdf', format='pdf')

logging.info("---------- show")

if es_entorno_interactivo:
    plt.show()

logging.info("---------- Guardamos modelo")


# Guardamos modelo ---------------------------------------- 

joblib.dump(best_xgb, 'files/modeling_output/model_fit/b01_best_xgb_model.joblib')

