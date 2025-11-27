import os  # <--- import 추가
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

# [수정] 환경변수에서 경로를 읽어오고, 없으면 기본값 사용
# 이제 docker-compose에서 넘겨준 값을 사용합니다.
PROJECT_ROOT_PATH = os.getenv("PROJECT_ROOT", "/data1/project/private")
PROJECT_DIR = f"{PROJECT_ROOT_PATH}/MLops"

default_args = {
    'owner': 'dhankim', # (이것도 os.getenv로 뺄 수 있음)
    'retries': 0,
    'retry_delay': timedelta(minutes=1),
}

with DAG(
    dag_id='yolo_automation_dhankim',
    default_args=default_args,
    description='YOLO MLOps Pipeline for dhankim',
    schedule_interval=None, 
    start_date=datetime(2024, 11, 27),
    catchup=False,
    tags=['mlops', 'yolo'],
) as dag:

    PROJECT_DIR = "/data1/project/private/MLops"

    # Task 1: 데이터 확인 (DVC Pull)
    # git 명령어 없이 cd로 이동 후 실행
    # Task 1
    pull_data = BashOperator(
        task_id='dvc_pull',
        # 변수 사용
        bash_command=f"cd {PROJECT_DIR} && dvc pull --force",
    )

    # Task 2: 학습 실행 (Train)
    train_model = BashOperator(
        task_id='train_yolo',
        bash_command=f"cd {PROJECT_DIR} && python script/train.py",
        env={
            'MLFLOW_TRACKING_URI': 'http://mlflow_server:5000',
            'GIT_PYTHON_REFRESH': 'quiet',
            'PATH': '/home/airflow/.local/bin:/usr/local/bin:/usr/bin:/bin'
        }
    )

    # 순서 연결
    pull_data >> train_model