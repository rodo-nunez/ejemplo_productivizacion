library(tidyverse)

# Leer datos --------------------------------------------------------------

data = arrow::read_feather("files/datasets/intermediate/a01_datos_preprocesados.feather")

# Filtrar datos -----------------------------------------------------------

periodo_de_interes = "201912"

data_filtrada =
  data |>
  filter(begin_date |>
           stringr::str_remove("-") |>
           stringr::str_sub(1, 6)  == periodo_de_interes)

# Guardar datos -----------------------------------------------------------

data_filtrada |> 
  arrow::write_feather("files/datasets/intermediate/a011_datos_filtrados.feather")
