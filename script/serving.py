from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import uvicorn
import os
import glob
import cv2
import numpy as np
from io import BytesIO
from PIL import Image

app = FastAPI(title="YOLO MLOps Inference Server")

# 전역 변수로 모델 관리
model = None

# 프로젝트 경로 (Docker 내부 경로 기준)
PROJECT_DIR = "/data1/project/private/MLops"
TRAIN_DIR = f"{PROJECT_DIR}/runs/train"

class ModelUpdate(BaseModel):
    model_path: str

def load_model(path: str):
    global model
    try:
        print(f"🔄 Loading model from: {path}")
        model = YOLO(path)
        print("✅ Model loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
def get_latest_model():
    """
    runs/train 폴더에서 가장 최근에 생성된 best.pt 파일을 찾습니다.
    """
    # exp_로 시작하는 모든 폴더 검색
    search_path = f"{TRAIN_DIR}/coco_*/weights/best.pt"
    list_of_files = glob.glob(search_path)
    
    if not list_of_files:
        return None
        
    # 생성 시간 순으로 정렬하여 가장 마지막 파일 선택
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

if initial_model_path:
    load_model(initial_model_path)

# 모델 재로딩 엔드포인트
@app.post("/reload")
def reload_model_endpoint(update: ModelUpdate):
    """
    Airflow로부터 새로운 모델 경로를 받아서 즉시 교체합니다.
    """
    if not os.path.exists(update.model_path):
        raise HTTPException(status_code=400, detail="Model file not found.")
    
    success = load_model(update.model_path)
    if success:
        return {"status": "success", "message": f"Model reloaded: {update.model_path}"}
    else:
        raise HTTPException(status_code=500, detail="Failed to load model.")   
 
# 전역 변수로 모델 로드 (서버 시작 시 1회)
print("🔄 Searching for the latest model...")
model_path = get_latest_model()

if model_path:
    print(f"✅ Model found: {model_path}")
    model = YOLO(model_path)
else:
    print("⚠️ No model found! Server will start but cannot predict.")
    model = None

@app.get("/")
def read_root():
    return {"status": "healthy", "model_path": model_path}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    이미지를 받아서 YOLO 추론 결과를 반환합니다.
    """
    if model is None:
        return {"error": "Model not loaded yet."}

    # 1. 이미지 읽기
    contents = await file.read()
    image = Image.open(BytesIO(contents))
    
    # 2. 추론 실행
    results = model.predict(image)
    
    # 3. 결과 파싱 (JSON 변환)
    detections = []
    for result in results:
        for box in result.boxes:
            detections.append({
                "class": int(box.cls),
                "name": model.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": box.xyxy.tolist()[0]
            })
            
    return {"filename": file.filename, "detections": detections}

if __name__ == "__main__":
    # 8000번 포트로 실행
    uvicorn.run(app, host="0.0.0.0", port=8888)