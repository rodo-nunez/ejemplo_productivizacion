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
