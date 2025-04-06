library(tidyverse)
library(reticulate)
library(glue)

# Setup de ambiente virtual y variables globales ---------------------------------------- 
reticulate::use_virtualenv("./.venv")
tryCatch(
  expr = {
    reticulate::source_python('params.py')
  },
  error = function(e){ 
      # (Optional)
      # Do this if an error is caught...
    print("Error encontrado, intentando de nuevo")
  }
)
reticulate::source_python('params.py')

# Leer datos --------------------------------------------------------------

entrenamiento_sufix = py$get_entrenamiento_sufix(py$bool_entrtenamiento_por_defecto) # TODO Lo correcto acá sería tomar un valor de la terminal, pero por tiempo, no implemente la demostración para esto. pero es totalmente posible

data = arrow::read_feather(glue("files/datasets/intermediate/a01_datos_preprocesados{entrenamiento_sufix}.feather"))

# Filtrar datos -----------------------------------------------------------

periodo_de_interes = "201912"

data_filtrada =
  data |>
  filter(begin_date |>
           stringr::str_remove("-") |>
           stringr::str_sub(1, 6)  == periodo_de_interes)

# Guardar datos -----------------------------------------------------------

data_filtrada |> 
  arrow::write_feather(glue("files/datasets/intermediate/a011_datos_filtrados{entrenamiento_sufix}.feather"))
