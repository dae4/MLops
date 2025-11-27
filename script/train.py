import os
import mlflow
from ultralytics import YOLO

def train_yolo():
    # ---------------------------------------------------------
    # 1. 환경 설정 (Local vs Docker)
    # ---------------------------------------------------------
    # Airflow(Docker)에서 실행될 때는 환경변수 'MLFLOW_TRACKING_URI'를 사용하고,
    # 내 컴퓨터에서 테스트할 때는 'http://localhost:5000'을 기본값으로 씁니다.
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    
    print(f"🔌 MLflow Tracking URI: {mlflow_uri}")
    
    # 실험 이름 설정 (MLflow UI 왼쪽에 표시될 이름)
    experiment_name = "YOLO_COCO128_pipeline"
    mlflow.set_experiment(experiment_name)

    # ---------------------------------------------------------
    # 2. 데이터 동기화 (DVC)
    # ---------------------------------------------------------
    # 데이터가 없으면 에러가 나므로, 안전하게 dvc pull을 한번 실행해줍니다.
    if os.path.exists("dvc.yaml") or os.path.exists("data.dvc"):
        print("📥 DVC 데이터 동기화 중...")
        exit_code = os.system("dvc pull")
        if exit_code != 0:
            print("⚠️ DVC Pull 실패 (데이터가 이미 있거나 설정 문제). 계속 진행합니다.")

    # ---------------------------------------------------------
    # 3. 학습 시작 및 기록
    # ---------------------------------------------------------
    with mlflow.start_run() as run:
        print(f"🚀 학습 시작! Run ID: {run.info.run_id}")
        
        # 태그 남기기
        mlflow.set_tag("model", "YOLOv11n")
        mlflow.set_tag("executor", "airflow" if os.getenv("AIRFLOW_HOME") else "local")

        # 모델 로드 (처음엔 자동으로 다운로드 됩니다)
        model = YOLO('yolo11n.pt') 

        # 학습 실행
        # data='data.yaml' : 아래 2번 단계에서 만들 설정 파일
        results = model.train(
            data='./script/coco128_custom.yaml',
            epochs=10,
            imgsz=640,
            batch=16,
            project="runs/train",
            name=f"coco_{run.info.run_id}",
            exist_ok=True,
            plots=True,
            # ★ 여기를 추가하세요!
            workers=0,  # 멀티프로세싱 끄기 (Deadlock 방지)
        )

        # ---------------------------------------------------------
        # 4. 결과 모델 업로드
        # ---------------------------------------------------------
        # Ultralytics는 학습 결과를 runs/train/실험명/weights/best.pt 에 저장합니다.
        best_model_path = str(results.save_dir / "weights" / "best.pt")
        
        if os.path.exists(best_model_path):
            print(f"💾 MLflow에 모델 업로드 중... ({best_model_path})")
            mlflow.log_artifact(best_model_path, artifact_path="model")
            mlflow.log_artifact(str(results.save_dir / "results.csv"), artifact_path="metrics")
        else:
            print("⚠️ 모델 파일을 찾을 수 없습니다.")

if __name__ == "__main__":
    train_yolo()