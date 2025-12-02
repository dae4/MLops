import os  
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import requests
import json
from airflow.operators.python import PythonOperator

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
    
    def notify_serving_server(**context):
        # 1. train_yolo 태스크가 반환한 값(모델 절대 경로)을 가져옴 (XCom)
        model_path = context['ti'].xcom_pull(task_ids='train_yolo')
        
        if not model_path:
            raise ValueError("No model path received from training task!")

        print(f"🚀 Sending reload request to Serving Server... Path: {model_path}")

        # 2. Serving 컨테이너의 내부 주소로 요청 전송
        # (Docker DNS 덕분에 'yolo_serving'이라는 이름으로 접속 가능)
        url = "http://yolo_serving:8000/reload"
        payload = {"model_path": model_path}
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status() # 에러 시 예외 발생
            print(f"✅ Reload Success: {response.json()}")
        except Exception as e:
            print(f"❌ Failed to reload model: {e}")
            raise
    
    reload_serving = PythonOperator(
        task_id='reload_serving',
        python_callable=notify_serving_server,
        provide_context=True # XCom 사용을 위해 필수
    )

    # 순서 연결: 데이터 -> 학습 -> 재로딩
    pull_data >> train_model >> reload_serving