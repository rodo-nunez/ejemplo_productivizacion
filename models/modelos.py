
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


