docker build -t caso_ejemplo_telecom:0.1.1 .
docker run --network=host -i --name caso_ejemplo_telecom caso_ejemplo_telecom:0.1.1
