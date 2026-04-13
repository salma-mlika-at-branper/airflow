FROM apache/airflow:2.10.5-python3.11

ARG CACHEBUST=1

USER root

# 1. Define where Rust lives so both Root and Airflow users can see it
ENV RUSTUP_HOME=/opt/rust/rustup \
    CARGO_HOME=/opt/rust/cargo \
    PATH="/opt/rust/cargo/bin:${PATH}"

# 2. Install dependencies and Rust in one clean step
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && mkdir -p /opt/rust \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path \
    && chmod -R a+w /opt/rust \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 3. Switch back to airflow user for the python packages
USER airflow


RUN pip install --upgrade pip
RUN pip install --no-cache-dir "protobuf==3.20.3" 
RUN pip install --no-cache-dir \
    transformers==5.0.0 \
    tokenizers==0.22.2 \
    torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124 \
    scikit-learn \
    pandas \
    sentencepiece \
    datasets \
    accelerate 
   