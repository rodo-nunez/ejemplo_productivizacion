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

expectation_monthly_charges_between = gx.expectations.ExpectColumnValuesToBeBetween(
    column="MonthlyCharges", max_value=60, min_value=0
)

# Ejemplos de multiples expectations y cómo parametrizarlas con runtime parameters ---------------------------------------- 

expectation_max_total_charges_between = gx.expectations.ExpectColumnMaxToBeBetween(
    column="TotalCharges",
    min_value={"$PARAMETER": "expect_total_charges_max_to_be_above"},
    max_value={"$PARAMETER": "expect_total_charges_max_to_be_below"},
)
expectation_paperless_billing_values = gx.expectations.ExpectColumnValuesToBeInSet(
    column="PaperlessBilling",
    value_set=("Yes", "No"),
    # value_set={"$PARAMETER": "expect_paperless_billing_values_to_be_in"}

)

runtime_expectation_parameters = {
    "expect_total_charges_max_to_be_above": 30,
    "expect_total_charges_max_to_be_below": 7000,
    "expect_paperless_billing_values_to_be_in": ("Yes", "No")
}
# TODO Aplicar estos parámetros al ejecutar

# Obtener el dataframe como un batch ----------------------------------------

df_contracts = pd.read_csv(
    "files/datasets/input/contract.csv"
)
df_contracts["TotalCharges"] = pd.to_numeric(df_contracts['TotalCharges'],errors="coerce")

batch_parameters = {"dataframe": df_contracts} # TODO: Agregar especificaciones de fecha a un batch

batch = batch_definition.get_batch(batch_parameters=batch_parameters)

# Testeamos la expectation ----------------------------------------
validation_results = batch.validate(expectation_monthly_charges_between)
print(validation_results)


# Testeando runtime expectations ---------------------------------------- 

validation_results = batch.validate(
    expectation_max_total_charges_between, expectation_parameters=runtime_expectation_parameters
)

print(validation_results)

# Validación de un batch, usando una validación ya definida ---------------------------------------- 

# TODO: Agregar definición de validación

# # Obtener definición de la validación
# validation_definition_name = "validation_definition_contract_dataframe"
# validation_definition = context.validation_definitions.get(validation_definition_name)

# #  Validar el dataframe ejecutando la validación sobre un batch definido
# validation_results = validation_definition.run(batch_parameters=batch_parameters)
# print(validation_results)