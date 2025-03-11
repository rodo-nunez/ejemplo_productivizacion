# Libraries ---------------------------------------- 
import great_expectations as gx
import pandas as pd

# Crear contexto y guardarlo como archivos ---------------------------------------- 
context = gx.get_context()
# context = context.convert_to_file_context() # Si guardamos esto como archivos, fallarán los siguientes comandos

print(context)

# Read data ---------------------------------------- 
df_contracts = pd.read_csv(
    "files/datasets/input/contract.csv"
)

# Connect to data and create a Batch ---------------------------------------- 
data_source = context.data_sources.add_pandas("pandas")
data_asset = data_source.add_dataframe_asset(name="pd dataframe asset contract")

batch_definition = data_asset.add_batch_definition_whole_dataframe("batch definition")
batch = batch_definition.get_batch(batch_parameters={"dataframe": df_contracts})

# Create an Expectation ---------------------------------------- 
expectation = gx.expectations.ExpectColumnValuesToBeBetween(
    column="MonthlyCharges", min_value=0, max_value=100
)

# Validate sample ---------------------------------------- 
validation_result = batch.validate(expectation)

# Define Data Source ---------------------------------------- 

# Define the Data Source's parameters:
# This path is relative to the `base_directory` of the Data Context.
source_folder = "./files/datasets/input"
data_source_name = "input_data"

# Create the Data Source:
data_source = context.data_sources.add_pandas_filesystem(
    name=data_source_name, base_directory=source_folder
)

# Define Data Asset ---------------------------------------- 

# data_source = context.data_sources.get(data_source_name)

# Define the Data Asset's parameters:
data_asset_name = "contract"

# Add the Data Asset to the Data Source:
file_csv_asset = data_source.add_csv_asset(name=data_asset_name)

# Save to file ---------------------------------------- 

context = context.convert_to_file_context() # Si guardamos esto como archivos, fallarán los siguientes comandos


