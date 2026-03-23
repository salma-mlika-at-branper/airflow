FROM apache/airflow:2.10.5-python3.11

ARG CACHEBUST=1

USER root
RUN apt-get update && apt-get install -y curl build-essential \
    && curl https://sh.rustup.rs -sSf | sh -s -- -y \
    && . "$HOME/.cargo/env"

USER airflow
RUN pip install --no-cache-dir "protobuf==3.20.3" 
RUN pip install --upgrade pip
RUN pip install --no-cache-dir \
    transformers==4.30.2 \
    torch \
    scikit-learn \
    pandas \
    sentencepiece
