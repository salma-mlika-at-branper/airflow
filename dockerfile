FROM apache/airflow:2.10.5


ARG CACHEBUST=1

USER airflow
RUN pip install --no-cache-dir \
    transformers==4.30.2 \
    torch \
    scikit-learn \
    pandas \
    sentencepiece