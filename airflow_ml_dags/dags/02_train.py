from datetime import timedelta

# The DAG object; we'll need this to instantiate a DAG
from airflow import DAG

# Operators; we need this to operate!
# from airflow.operators.bash import BashOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from airflow.models import Variable
from airflow.sensors.filesystem import FileSensor

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
    dag_id="02_train",
    default_args=default_args,
    schedule_interval="@weekly",
    start_date=days_ago(14),
    description="Training model",
) as dag:
    # waiting_for_data_file = FileSensor(
    #     task_id='waiting_for_data_file',
    #     fs_conn_id=f"{DATA_DIR}raw/{{ ds }}/data.csv",
    #     # filepath="/data/raw/{{ ds }}/data.csv",
    #     # filepath="raw/{{ ds }}/data.csv",
    #
    #     poke_interval=30
    # )
    #
    # waiting_for_target_file = FileSensor(
    #     task_id='waiting_for_target_file',
    #     fs_conn_id=f"{DATA_DIR}raw/{{ ds }}/target.csv",
    #     # fs_conn_id=DATA_DIR,
    #     # filepath="/data/raw/{{ ds }}/target.csv",
    #     # filepath="raw/{{ ds }}/target.csv",
    #     poke_interval=30
    # )

    preprocess = DockerOperator(
        image="airflow-preprocess",
        command="--input-dir /data/raw/{{ ds }} --output-dir /data/processed/{{ ds }}",
        task_id="docker-airflow-preprocess",
        do_xcom_push=False,
        volumes=[f"{DATA_DIR}:/data"]
    )

    split = DockerOperator(
        image="airflow-split",
        command="--input-dir /data/processed/{{ ds }} --output-dir /data/split/{{ ds }}",
        task_id="docker-airflow-split",
        do_xcom_push=False,
        volumes=[f"{DATA_DIR}:/data"]
    )

    train = DockerOperator(
        image="airflow-train",
        command="--input-dir /data/split/{{ ds }} --output-dir /data/models/{{ ds }}",
        task_id="docker-airflow-train",
        do_xcom_push=False,
        volumes=[f"{DATA_DIR}:/data"]
    )

    validate = DockerOperator(
        image="airflow-validate",
        command="--input-dir /data/split/{{ ds }} --models-dir /data/models/{{ ds }}",
        task_id="docker-airflow-validate",
        do_xcom_push=False,
        volumes=[f"{DATA_DIR}:/data"]
    )

    # [waiting_for_data_file, waiting_for_target_file] >> preprocess >> split >> train >> validate
    preprocess >> split >> train >> validate
