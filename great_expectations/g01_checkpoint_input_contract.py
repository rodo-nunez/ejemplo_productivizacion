# Librerias y leer contexto ---------------------------------------- 
import great_expectations as gx
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

# Leer datos ---------------------------------------- 

df_contracts = pd.read_csv(
    "files/datasets/input/contract.csv"
)
df_contracts["TotalCharges"] = pd.to_numeric(df_contracts['TotalCharges'],errors="coerce")

# Definiciones GX ---------------------------------------- 

context = gx.get_context()

checkpoint_name = "input_dataframes_checkpoint"
checkpoint = context.checkpoints.get(checkpoint_name)
batch_parameters = {"dataframe": df_contracts}
runtime_expectation_parameters = {
    "expect_total_charges_max_to_be_above": 30,
    "expect_total_charges_max_to_be_below": 7000,
    "expect_paperless_billing_values_to_be_in": ("Yes", "No")
}

# Checkpoint ---------------------------------------- 

validation_results = checkpoint.run(
    batch_parameters=batch_parameters, expectation_parameters=runtime_expectation_parameters
)