from datetime import timedelta

# The DAG object; we'll need this to instantiate a DAG
from airflow import DAG

# Operators; we need this to operate!
# from airflow.operators.bash import BashOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from airflow.models import Variable

DATA_DIR = Variable.get("DATA_DIR")
MODEL_DIR = Variable.get("MODEL_DIR")


# These args will get passed on to each operator
# You can override them on a per-task basis during operator initialization
default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="03_predict",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=days_ago(14),
    description="Predicting with model",
) as dag:
    preprocess = DockerOperator(
        image="airflow-preprocess",
        command="--input-dir /data/raw/{{ ds }} --output-dir /data/processed/{{ ds }}",
        task_id="docker-airflow-preprocess",
        do_xcom_push=False,
        volumes=[f"{DATA_DIR}:/data"]
    )

    predict = DockerOperator(
        image="airflow-predict",
        command="--input-dir /data/processed/{{ ds }} "
                "--output-dir /data/predicted/{{ ds }} "                
                # "--models-dir /data/models/{{ ds }}",
                f"--models-dir {MODEL_DIR}",
        task_id="docker-airflow-predict",
        do_xcom_push=False,
        volumes=[f"{DATA_DIR}:/data"]
    )

    preprocess >> predict
