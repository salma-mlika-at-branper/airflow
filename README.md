
unzip airflow-project.zip; 
bash ./Miniconda3-latest-Linux-x86_64.sh

conda create -n airflow_project python=3.12 -c conda-forge -y
# airflow

# SSH connect to deployment server

eval `ssh-agent -s`
ssh-add /root/.ssh/id_ed25519

ssh ubuntu@10.142.0.3 -i /root/.ssh/id_ed25519
