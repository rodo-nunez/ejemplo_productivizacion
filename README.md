# Canvas Analytics: **Nombre del Proyecto**

## Recursos Relevantes
- [Documenentos](URL) : Link va dirigido a la documentacion del proyecto
- [Datasets](URL) : Link al catalogo de tabla con informacion de los inputs.
- [Dashboard](URL) : Link al dashboard.
- [Seguimiento de Modelo](URL) : Link al Seguimiento de Modelo.
- [Grupo en Slack, Teams u otro](URL): Link al grupo o canales de comunicación.

## ¿Cuál es la “Pregunta Clave”?

Pregunta de negocio que se requiere contestar con este proyecto. Esta es una pregunta general de negocio y usualmente el proyecto desarrollará solo una (o algunas) de todas las posibles soluciones.

## Objetivos del negocio

Indica cuáles son las cosas que el negocio busca al implementar este proyecto.

  + Objetivo N1
  + Objetivo N2

## Breve descripción de la iniciativa

Descripción general de lo que se va a hacer para contestar la pregunta clave y cumplir con el objetivo de negocio. Debe ser preciso y fácil de entender

## Alcance

Definiciones que acotan el problema, como por ejemplo, los segmentos (productos, personas, zonas) a considerar, historia de datos, etc.

## Entregable

Definición del producto final que requiere la contraparte. Por ejemplo: un score continuo, una decisión binaria, un reporte, todo lo anterior, etc.

## Criterios de satisfacción y cómo se medirá el éxito de la iniciativa

Métricas relevantes para el negocio, su valor actual y la meta a alcanzar ya sea en valor, porcentaje de aumento/disminución o cambios que se espera generar con la ejecución del proyecto. Breve explicación de cómo se medirá la métrica definida

**Criterio Negocio**

 + Criterio n1 + medición
 + Criterio n2 + medición

**Criterio Analytics**

 + Criterio n1 + medición
 + Criterio n2 + medición
 
## Criterio de éxito MVP

Punto de corte para el cual se considerará que el trabajo realizado es lo suficientemente bueno / predictivo / robusto para poner el proyecto en ejecución. Puede definirse durante la etapa MVP.


## Potencial económico de la iniciativa

Valor estimado anual en millones de pesos que generará el proyecto y una breve explicación de cómo se obtiene este valor.

## Involucrados

### Analytics

Listado de todas las personas que participan del proyecto, incluyendo al responsable y el coordinador.

  + Persona n1 (Periodo)
  + Persona n2 (Periodo)

### Unidades de negocio

Listado de todas las personas de otras áreas involucradas en el proyecto, indicando el área de negocio en la cual trabajan, su nombre y rol en el proyecto.

   + Persona n1 (Periodo)
   + Persona n2 (Periodo)


## Bitacora

### 1. Ideación  [ESTADO] 
  + **Descripción**:
  + **Periodo**:

### 2. MVP [ESTADO]  
  + **Descripción**:
  + **Periodo**:

### 3. Implementación - Primer Piloto [ESTADO] 
  + **Descripción**:
  + **Periodo**:

### 4. Implementación - Plan Operativo [ESTADO] 
  + **Descripción**:
  + **Periodo**:

### 5. Productivización [ESTADO] 
  + **Descripción**:
  + **Periodo**:

# Estructura de proyecto

Basicamente el proyecto se divide en 2 grande mundos, lo que esta dentro de la carpeta "files" y lo que no esta.
Esta división es necesaria porque lo que esta fuera de files es codigo que se respalda en Git (junto con todos los archivos de texto plano que no sean datasets u otro tipo de archivo grande), mientras que los archivos que no son texto plano o son muy grandes para ser versionados en Git, van a la carpeta files que se respalda en Amazon S3.

Cada compomente de un proyecto ha sido mapeado a una carpeta del template, de ser necesario mas, se pueden crear. La estructura es la siguiente:

## Fuera de Files

Todo este contenido se respalda con git, por lo que debe haber solo codigo. Todo el resto debe estar considerado en el `.gitignore`.

+ **sandbox**: Scripts que no son parte del proyecto, son experimentos o no es claro donde ponerlos a priori.
+ **reports**: Scripts para generar reportes, pero no el reporte en si. Los reportes van en files/documentation/reports
+ **preprocesing**: Scripts para preprocesar y limpiar datos que seran utilizados en los modelos. Los input y output de estos procesos deben estar en files/dataset (descrito en detalle mas adelante)
+ **models**: Scripts que generan los modelos, los modelos deben ser guardados en files/modeling_output/model_fit
+ **exploratory**: Analisis exploratorios o descriptivos que aportan valor, los resultados van en files/modeling_output/reportes
+ **function**: Funciones del proyecto, estos archivos deben tener el mismo nombre que la funcion. Las funciones son cargadas en el `init.R` 
+ **pipeline**: Scripts para ejecutar de forma continua todos los scripts involucrados para alguna seccion del proyecto, como preproceso o preproceso + modelado o ejecucion, etc.
+ **execution**: Scripts para ejecutar el modelo o proyecto. Estos son los que se llamarían desde un script en pipeline para el ejecutar con una periodicidad dada.
+ **test**: Scripts con tests unitarios para checkear que no haya problemas en varios niveles del codigo. Puede haber subcarpetas para guardar los test de preproceso, de modelo, de ETL o de lo que sea necesario por separado. Tambien se pueden separar por lenguaje de programacion si es necesario, para facilitar la ejecicion de todos los test por carpetas.

## Files

Este contenido se respalda en algún object storage (como S3 en el caso de AWS).

* **files/datasets**: Son los datos utilizados para el proyecto, las 3 carpetas que estan adentro corresponden al siguiente flujo:
    + files/datasets/**input**: Archivos entregados por el cliente u otra persona que van a ser usados para modelar, pero aun necesitan procesamiento.
    + files/datasets/**intermediate**: Archivos ya procesados que seran utilizados para crear datasets que seran utilizados para modelar en una ultima instancia, pero aun no estan listos. Usar tipos de archivos que preserven formatos, no como `.csv`.
    + files/datasets/**output**: Datasets listos para modelar en formato feather, RDS, etc.
    

* **files/documentation** : Son los documentos relacionados con el proyecto, pero que no contienen datos para modelar, como papers, presentaciones, etc. Todo deberia estar en un object storage o algo similar.
    + [ObjectStorage](URL)/Proyectos/NOMBRE_DEL_PROYECTO/**docs_cliente**: Documentos entregados por la contraparte & papers.
    + [ObjectStorage](URL)/Proyectos/NOMBRE_DEL_PROYECTO/**hitos**: Documentacion que respaldan los cierre de los hitos.
    + [ObjectStorage](URL)/Proyectos/NOMBRE_DEL_PROYECTO/**minutas**: Respaldo de todas las minutas del proyecto.
    + files/documentation/**reportes**: Reportes de performance de modelos, avances o presentaciones propias. Aqui pueden ir los output html/pdf/pptx/md/etc de reportes hechos en Markdown, RMarkdown, Notebook, LaTeX, PowerPoint o lo que sea.
    
    
* **files/modeling_output/** Aca se guardan los resultados del trabajo en los proyectos:
    + files/modeling_output/**figures** : Graficos o cualquier otro tipo de archivo jpg/png/pdf/etc con imagener, graficos, tablas o lo que uno quiera mostrar y guardar.
    + files/modeling_output/**model_fit**: En esta carpeta se guardan los objetos con los modelos, ya sea un RDS o un h2o
    + files/modeling_output/**logs**: Aqui deben guardarse todos los logs de tiempos de ejecusion, errores de codigos automaticos, logs de debugging puntuales y otros tipos de logs.
    + files/modeling_output/**results**: Aqui deben guardarse todos los archivos con resultados importantes del proyecto y/o modelo. Por ejemplo, las tablas resumenes de performance de cada iteracion, importancia de las variables y cualquier otro tipo de resultado del modelo que sea importante rescatar. 

## Nombres de archivos

Para mantener una estructura consistente y poder encontrar facilmente el resultado de la ejecusion que uno busca o el modelo que se entreno con ciertos parametros, etc, es importante que los nombres de los archivos de salida de cada script tengan consistencia. Por eso, es recomendable crear una funcion que asigne nombres a los archivos pasandole una serie de parametros. Estos parametros pueden ser las variables mas importantes que se modifican en el modelo, la fecha o periodo de ejecucion, el codigo del script que escribio el archivo, etc. Cada proyecto debe tener su propia funcion y en algunos casos bastara con 4 parametros, en otros se necesitarán 20.

Uno podría usar librerías que hagan algo así por uno. Independiente del método, es bueno tener consistencia para que todo sea más fácil de encontrar y entender.

# Ejecusión del proyecto

## Set up de ambiente virtual 

Para ejecutar este proyecto de forma reproducible, es importante tener ambientes virtuales bien definidos y documentados.

En el caso de este proyecto, usamos la versión de Python `3.12.7`, con las librerías en el `requirements.txt`.

Para replicar el ambiente virtual, simplemente ejecuta

```sh
python -m venv .venv
```

Luego, selecciona el ambiente virtual (o actívalo). Si estás usando Visual Studio Code, puedes seleccionarlo como tu ambiente por defecto con el pop up que aparece al crearlo, o usando el comando `Python: Select Interpreter` al abrir la paleta de comandos. 

Si no usas VSC, sino que lo quieres hacer por terminal, basta con ejecutar en una terminal de tipo Bash

```sh
source .venv/Scripts/activate
```

## Ejecución del proyecto

### Entrenamiento

Para ejecutar el pipeline de entrenamiento, ejecutar

```sh
python pipeline/p01_pipeline.py --modo_prueba False
```
