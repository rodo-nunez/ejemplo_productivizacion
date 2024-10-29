# Reemplazo de nulos ----------------------------------------

def replace_spaces(DataFrames):
    for df in DataFrames:
        df.replace(' ', np.nan, inplace=True)


dataframes = [df_contract, df_personal, df_internet, df_phone]
replace_spaces(dataframes)

df_contract['TotalCharges'] = df_contract['TotalCharges'].fillna(0)


# Escalamiento ---------------------------------------- 

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