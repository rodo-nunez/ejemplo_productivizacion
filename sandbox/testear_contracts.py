# Librerias y leer contexto ---------------------------------------- 
import great_expectations as gx
import pandas as pd

context = gx.get_context()

# Si usasemos un CSVAsset ---------------------------------------- 

# data_source_name = "input_data"
# data_asset_name = "contract"

# data_asset = context.data_sources.get(data_source_name).get_asset(data_asset_name)

# Usando un dataframe asset ---------------------------------------- 

data_source_name = "pandas"
data_asset_name = "pd dataframe asset contract"

data_asset = context.data_sources.get(data_source_name).get_asset(data_asset_name)

batch_definition_name = "contact_batch"

# Agregamos batch definition al data asset ---------------------------------------- 

# batch_definition = data_asset.add_batch_definition_whole_dataframe(
#     batch_definition_name
# )

# Ejemplo de lectura de batch definition que ya estaba creada en el data asset ---------------------------------------- 

batch_definition = (
    context.data_sources.get(
        data_source_name).get_asset(
            data_asset_name).get_batch_definition(
                batch_definition_name)
)

# Creamos una expectation para testear ----------------------------------------

expectation = gx.expectations.ExpectColumnValuesToBeBetween(
    column="MonthlyCharges", max_value=60, min_value=0
)

# Obtener el dataframe como un batch ----------------------------------------

df_contracts = pd.read_csv(
    "files/datasets/input/contract.csv"
)

batch_parameters = {"dataframe": df_contracts}

batch = batch_definition.get_batch(batch_parameters=batch_parameters)

# Testeamos la expectation ----------------------------------------
validation_results = batch.validate(expectation)
print(validation_results)


# Validación de un batch, usando una validación ya definida ---------------------------------------- 

# TODO: Agregar definición de validación

# # Obtener definición de la validación
# validation_definition_name = "validation_definition_contract_dataframe"
# validation_definition = context.validation_definitions.get(validation_definition_name)

# #  Validar el dataframe ejecutando la validación sobre un batch definido
# validation_results = validation_definition.run(batch_parameters=batch_parameters)
# print(validation_results)