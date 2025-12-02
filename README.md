# 🚀 YOLO Object Detection MLOps Pipeline (Single-Node)

본 문서는 **NVIDIA GPU 환경**에서 **Docker Compose**를 활용하여 데이터 관리(DVC), 실험 추적(MLflow), 자동화(Airflow)를 통합하는 파이프라인 구축 매뉴얼입니다.

---
Phase 1	인프라 구축	Docker Compose, GPU 연결, DB(Postgres) 연동	✅ 완료
Phase 2	데이터 관리	DVC 설치, 대용량 스토리지(/data2) 연결, Symlink 최적화	✅ 완료
Phase 3	학습 파이프라인	Airflow DAG 작성, MLflow 실험 기록, 자동화 구현	✅ 완료
Phase 4	모델 배포 (Serving)	FastAPI 추론 서버 구축, Docker 통합	👈 Next Step
Phase 5	운영 고도화	모델 레지스트리 관리, 모니터링(Drift 감지)	⬜ 예정
---

## 1. 초기 환경 설정 (Host Setup)

### 1.1. 환경 변수 파일 (.env) 생성

프로젝트 루트 디렉토리에 `.env` 파일을 생성하고, 서버의 **실제 경로**에 맞게 수정합니다.

```ini
# .env

# 프로젝트 메인 경로 (절대 경로: /data1/project/private)
PROJECT_ROOT=/data1/project/private

# DVC 캐시를 저장할 대용량 디스크 경로
DATA_STORAGE=/data2

# Airflow 접속 정보
MY_USER=dhankim
MY_EMAIL=dhankim@example.com

```

1.2. 필수 도구 설치 및 권한 부여
Docker가 GPU에 접근하고, Airflow가 파일에 쓸 수 있도록 설정합니다. (Linux Host 터미널에서 실행)

``` Bash

# 1. NVIDIA Container Toolkit 설치 (GPU와 Docker 연결)
# (이미 설치되어 있다면 생략)
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 2. DVC 캐시 폴더 생성 및 권한 대개방
mkdir -p $DATA_STORAGE/dvc-storage
sudo chmod -R 777 $DATA_STORAGE 
sudo chmod -R 777 ${PROJECT_ROOT}
```

## 2. DVC 데이터 저장소 설정
데이터 용량을 절약하고 Docker 환경과 호환되도록 심볼릭 링크(Symlink) 전략을 사용합니다.

```Bash

# Host 터미널 (프로젝트 루트에서 실행)
cd ${PROJECT_ROOT}/MLops

# 1. DVC 초기 설정 (캐시 경로를 /data2로 지정)
dvc config cache.dir $DATA_STORAGE/dvc-storage
dvc config cache.type symlink
dvc config cache.shared group

# 2. 데이터 다운로드 및 등록 (예: COCO128)
# (데이터를 다운받아 dataset/dvc-storage 폴더에 넣었다고 가정)
dvc add dataset/dvc-storage/coco128
git add .dvc/config dataset/dvc-storage/coco128.dvc
git commit -m "Initial DVC setup and COCO128 data tracking"
```

# 3. Docker Compose (Infrastructure)

## 3.1. docker-compose.yml
핵심: DB 안정성(Postgres), GPU 할당, OpenCV 자동 수정 (entrypoint), 계정 자동 생성.

# 4. MLflow, airflow 구축

# 5. 실행 및 사용 (Execution)
## 5.1. 서비스 시작

```bash 
docker compose up -d --force-recreate
```

## 5.2. 접속 및 실행

1. Airflow (Web UI): http://localhost:8088 (로그인: ${MY_USER} / ${MY_USER})
2. MLflow (UI): http://localhost:5000
3. 파이프라인 실행: Airflow에서 yolo_automation_dhankim Unpause 후 Trigger.