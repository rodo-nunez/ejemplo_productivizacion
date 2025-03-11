FROM ubuntu:22.04

ENV WORKDIR=caso_ejemplo_telecom
ENV DEBIAN_FRONTEND=noninteractive
ENV python_version_1=3
ENV python_version_2=12
ENV python_version_3=7
ENV python_version=${python_version_1}.${python_version_2}.${python_version_3}
ENV python_version_1_2=${python_version_1}.${python_version_2}

# Instalacion varias ---------------------------------------- 
RUN apt-get update && apt-get install -y sudo
RUN sudo apt-get install software-properties-common dirmngr -y
RUN sudo apt install wget -y
RUN apt-get install nano
RUN apt-get install -y lsb-release && apt-get clean all

# Instalacion y setup de locales ---------------------------------------- 
RUN sudo apt-get update
RUN sudo apt-get install locales
RUN sudo locale-gen es_ES.UTF-8
RUN sudo dpkg-reconfigure locales

# Instalacion R ---------------------------------------- 
RUN sudo wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc
RUN sudo add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/" -y
RUN echo "2 85" | apt-get install r-base -y 
RUN sudo apt install -y -V ca-certificates lsb-release wget
RUN sudo wget https://apache.jfrog.io/artifactory/arrow/$(lsb_release --id --short | tr 'A-Z' 'a-z')/apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb
RUN sudo apt install -y -V ./apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb
RUN sudo apt update
RUN sudo apt-get install build-essential libssl-dev libfontconfig1-dev libcurl4-openssl-dev libgit2-dev unixodbc-dev libxml2-dev zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libreadline-dev libffi-dev libsqlite3-dev libbz2-dev wget libharfbuzz-dev libfribidi-dev libfreetype6-dev libpng-dev libtiff5-dev libjpeg-dev libcairo2-dev cmake libmagick++-dev libarrow-dev libarrow-dataset-dev libarrow-flight-dev libarrow-glib-dev gfortran -y
RUN sudo apt-get install apt-transport-https ca-certificates curl gnupg lsb-release -y

# Instalacion de docker ---------------------------------------- 
RUN sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg |  gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
RUN echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

RUN apt-get update && sudo apt-get install docker-ce docker-ce-cli containerd.io -y

RUN sudo apt-get install docker-ce containerd.io docker-ce-cli -y 
RUN sudo apt-get install libgirepository1.0-dev build-essential software-properties-common apt-transport-https ca-certificates pkg-config libcairo2-dev libdbus-glib-1-dev openjdk-17-jdk openjdk-17-jre -y

# Instalacion Java y RJava ---------------------------------------- 

RUN sudo apt-get install openjdk-17-jdk openjdk-17-jre r-cran-rjava -y
RUN sudo R CMD javareconf JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64/
RUN sudo Rscript -e "install.packages('rJava')"

# Instalacion python ---------------------------------------- 
RUN sudo apt-get install -y build-essential \
  zlib1g-dev \
  libncurses5-dev \
  libgdbm-dev \
  libnss3-dev \
  libssl-dev \
  libreadline-dev \
  libffi-dev \
  wget \
  libsqlite3-dev \
  libbz2-dev
RUN sudo wget https://www.python.org/ftp/python/${python_version}/Python-${python_version}.tgz && \
  sudo tar -xf Python-${python_version}.tgz && \
  cd Python-${python_version} && \
  ./configure --enable-optimizations && \
  make -j $(nproc) && \
  make altinstall
RUN export PATH=/usr/bin/python3:$PATH
RUN echo alias="python=/usr/local/bin/python${python_version_1_2}" >> ~/.bashrc
RUN apt-get install python3-pip -y

# Copiar elementos básicos ---------------------------------------- 

# COPY ./instalar_R.sh /instalar_R.sh
# RUN sudo sh /instalar_R.sh

WORKDIR "/${WORKDIR}"
COPY requirements.txt requirements.txt
COPY renv.lock renv.lock
# COPY renv renv

# Copy .env ---------------------------------------- 
# ! Copiar el .env solo en caso de pruebas y tener mucho cuidado con filtrar credenciales

# RUN cp .env ~/.env 

# Setup ambiente viertual python ---------------------------------------- 

ENV VIRTUAL_ENV=/opt/venv
RUN python${python_version_1_2} -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip${python_version_1_2} install --upgrade pip
RUN pip${python_version_1_2} install -r requirements.txt

EXPOSE 443

# Setup de ambiente virtual R ---------------------------------------- 

RUN R -e "install.packages('renv')"
RUN R -e "renv::restore()"

# Copiar proyecto y otros ---------------------------------------- 

COPY . .

# Ejecutar proyecto ---------------------------------------- 

ENTRYPOINT [ "sh", "docker/docker_setup_and_run.sh" ] 
CMD [ "--modo_prueba", "False", "--bool_entrtenamiento", "False", "--periodo", "202408" ]
