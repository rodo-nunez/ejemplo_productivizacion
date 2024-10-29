
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
