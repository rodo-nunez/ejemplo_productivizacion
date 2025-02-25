# Definicion de variables importantes ---------------------------------------- 

# * A veces es importante definir variables como 
# - A que IP nos conectaremos. Esa IP podria ir cambiando y habría que setearla en cada conexión
# - Formateo de parametros, en caso de que no sea un simple "today" y listo
# - Testear conexión a IP, para asegurarnos de que estamos llegando a la base de datos que usamos. Si no, lanzar un error.
# - Otros

# Ejecusion de pipeline usando comandos de docker ---------------------------------------- 

echo Ejecutando p01 con parametros: $@

python pipeline/p01_pipeline.py $@

# python pipeline/p01_pipeline.py --modo_prueba "False" --bool_entrtenamiento "False" --periodo "202408" 