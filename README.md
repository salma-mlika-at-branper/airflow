
```bash

unzip airflow-project.zip; 
bash ./Miniconda3-latest-Linux-x86_64.sh

conda create -n airflow_project python=3.12 -c conda-forge -y
```

# airflow

# SSH connect to deployment server

```bash
eval `ssh-agent -s`
ssh-add /root/.ssh/id_ed25519

ssh ubuntu@10.142.0.3 -i /root/.ssh/id_ed25519


cd ~/Downloads/airflow

conda env list
conda activate airflow_project

sudo docker build ./. --tag airflow-project:latest --file ./dockerfile

sudo docker compose -f docker-compose.yaml up -d
```

```bash
eval `ssh-agent -s`
ssh-add /root/.ssh/id_ed25519

ssh ubuntu@10.132.0.2 -i /root/.ssh/id_ed25519

# after connecting to ssh
sudo -s

cd /media/sentinel-nvidia-t4/ubuntu/airflow
```

```bash
docker compose up -d
docker compose down
docker build -t my-airflow:latest .

# create a databese:
docker compose run --rm airflow-scheduler \
    airflow users create \
    --username admin \
    --firstname Salma \
    --lastname Admin \
    --role Admin \
    --email admin@example.com \
    --password admin
initialize the database:
    docker compose run --rm airflow-scheduler airflow db init'''
```
