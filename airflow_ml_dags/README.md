
чтобы развернуть airflow, предварительно собрав контейнеры
~~~
# для корректной работы с переменными, созданными из UI
export FERNET_KEY=$(python -c "from cryptography.fernet import Fernet; FERNET_KEY = Fernet.generate_key().decode(); print(FERNET_KEY)")
export DATA_PATH='your_data_path'
export MODEL_PATH='your_model_path'
# также эти переменные можно задать из GUI позже

# для работы
docker-compose up --build
# или
sudo -E docker-compose up --build
~~~

Самооценка:
* (+) Поднимите airflow локально, используя docker compose (можно использовать из примера https://github.com/made-ml-in-prod-2021/airflow-examples/) 
1) (+5) Реализуйте dag, который генерирует данные для обучения модели, данные взял из sklearn
2) (+10) Реализуйте dag, который обучает модель еженедельно, используя данные за текущий день. 
3) (+5) Реализуйте dag, который использует модель ежедневно 
    * (0) Реализуйте сенсоры на то, что данные готовы для дагов тренировки и обучения (3 доп балла)
4) (+10) вы можете выбрать 2 пути для выполнения ДЗ. Всё через docker_operator.
5) (0) Протестируйте ваши даги (5 баллов) 
6) (0) В docker compose так же настройте поднятие mlflow (5 доп баллов)
7) (0) вместо пути в airflow variables  используйте апи Mlflow Model Registry (5 доп баллов)
8) (0) Настройте alert в случае падения дага (3 доп. балла)
9) (+1)традиционно, самооценка (1 балл)

Предполагаемая оценка 31 * 0.6 == 18.6 баллов