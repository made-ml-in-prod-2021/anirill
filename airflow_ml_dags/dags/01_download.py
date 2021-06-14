from datetime import timedelta

# The DAG object; we'll need this to instantiate a DAG
from airflow import DAG

# Operators; we need this to operate!
# from airflow.operators.bash import BashOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from airflow.models import Variable

DATA_DIR = Variable.get("DATA_DIR")

# These args will get passed on to each operator
# You can override them on a per-task basis during operator initialization
default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="01_download",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=days_ago(14),
    description="Getting and data",
) as dag:
    download = DockerOperator(
        image="airflow-download",
        command="/data/raw/{{ ds }}",
        network_mode="bridge",
        task_id="docker-airflow-download",
        do_xcom_push=False,
        # !!! HOST folder(NOT IN CONTAINER) replace with yours !!!
        volumes=[f"{DATA_DIR}:/data"]
    )

    # preprocess = DockerOperator(
    #     image="airflow-preprocess",
    #     command="--input-dir /data/raw/{{ ds }} --output-dir /data/processed/{{ ds }}",
    #     task_id="docker-airflow-preprocess",
    #     do_xcom_push=False,
    #     volumes=["/Users/mikhail.maryufich/PycharmProjects/airflow_examples/data:/data"]
    # )
    #
    # predict = DockerOperator(
    #     image="airflow-predict",
    #     command="--input-dir /data/processed/{{ ds }} --output-dir /data/predicted/{{ ds }}",
    #     task_id="docker-airflow-predict",
    #     do_xcom_push=False,
    #     volumes=["/Users/mikhail.maryufich/PycharmProjects/airflow_examples/data:/data"]
    # )

    download
    # download >> preprocess >> predict
