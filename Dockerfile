FROM python:3.10.6

ENV WORKDIR=caso_ejemplo_telecom
ENV python_version_1=3
ENV python_version_2=10
ENV python_version_3=6
ENV python_version=${python_version_1}.${python_version_2}.${python_version_3}
ENV python_version_1_2=${python_version_1}.${python_version_2}

# Instalar dependencias del sistema y Rust ----------------------------------------

    RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Añadir Rust y Cargo al PATH ----------------------------------------
    
ENV PATH="/root/.cargo/bin:${PATH}"

# Copiar elementos básicos ---------------------------------------- 

WORKDIR "/${WORKDIR}"
COPY requirements.txt requirements.txt
COPY renv.lock renv.lock

# Copy .env ---------------------------------------- 
# ! El .env hay que copiarlo en el docker del proyecto, no la plantilla generica. Y solo en caso de tener credenciales productivas en un ambiente de producción privado

# RUN cp .env ~/.env 

# Setup ambiente viertual python ---------------------------------------- 

ENV VIRTUAL_ENV=/opt/venv
RUN python${python_version_1_2} -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip${python_version_1_2} install --upgrade pip
RUN pip${python_version_1_2} install -r requirements.txt

EXPOSE 443
EXPOSE 1522

# Copiar proyecto y otros ---------------------------------------- 

COPY . .

# Setup de ambiente virtual R ---------------------------------------- 

# RUN R -e "renv::restore()"
# RUN Rscript load_packages.R

# Ejecutar proyecto ---------------------------------------- 

# RUN sh pipeline/p01_pipeline.sh
