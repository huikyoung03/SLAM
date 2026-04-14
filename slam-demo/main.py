import os
import time
import json
import threading     # <--- 🚨 이거 추가!
import subprocess
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

app = FastAPI(title="SLAM Sensor & Vision Streamer")

# CORS (통신 에러 방지)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 절대 경로 설정 (에러 방지)
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
STATIC_DIR = BASE_DIR / "static"

UPLOAD_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

# 정적 파일 연결
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ======== 🏠 대문 (HTML 안전하게 넘겨주기) ========
@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = STATIC_DIR / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h1>에러!</h1><p>static 폴더 안에 index.html 파일이 없습니다.</p>")
    
    # utf-8로 강제 지정하여 읽기 (글자 깨짐 방지)
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# ======== 📸 1. 이미지 프레임 수신 (HTTP POST) ========
@app.post("/stream-frame")
async def stream_frame(
    session_id: str = Form(...),
    timestamp: float = Form(...),
    frame: UploadFile = File(...)
):
    session_dir = UPLOAD_DIR / session_id
    rgb_dir = session_dir / "rgb"
    
    if not session_dir.exists():
        rgb_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "rgb.txt").write_text("# timestamp filename\n")
        (session_dir / "imu.txt").write_text("# timestamp gx gy gz ax ay az\n")

    filename = f"{timestamp:.6f}.jpg"
    file_path = rgb_dir / filename
    
    content = await frame.read()
    with open(file_path, "wb") as f:
        f.write(content)

    with open(session_dir / "rgb.txt", "a") as f:
        f.write(f"{timestamp:.6f} rgb/{filename}\n")

    return {"status": "ok"}


# ======== 🌐 2. IMU 센서 수신 (WebSocket) ========
@app.websocket("/ws/imu")
async def websocket_imu(websocket: WebSocket):
    await websocket.accept()
    print("📱 모바일 IMU 웹소켓 연결됨!")
    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            
            if data.get("type") == "imu":
                session_id = data.get("session_id")
                ts_sec = data.get("timestamp") / 1000.0 
                
                session_dir = UPLOAD_DIR / session_id
                
                if not session_dir.exists():
                    (session_dir / "rgb").mkdir(parents=True, exist_ok=True)
                    (session_dir / "rgb.txt").write_text("# timestamp filename\n")
                    (session_dir / "imu.txt").write_text("# timestamp gx gy gz ax ay az\n")

                accel = data.get("accel_g", {})
                gyro = data.get("gyro", {})
                
                ax, ay, az = accel.get("x", 0), accel.get("y", 0), accel.get("z", 0)
                ga, gb, gc = gyro.get("alpha", 0), gyro.get("beta", 0), gyro.get("gamma", 0)
                
                with open(session_dir / "imu.txt", "a") as f:
                    f.write(f"{ts_sec:.6f} {ga} {gb} {gc} {ax} {ay} {az}\n")

    except WebSocketDisconnect:
        print("📱 모바일 IMU 연결 종료")
        
def run_orb_slam3(session_id: str):
    print(f"🚀 [SLAM 엔진 가동] 세션 {session_id} 분석을 시작합니다...")
    
    # 데이터가 저장된 폴더 경로
    dataset_path = UPLOAD_DIR / session_id
    
    
    # 카메라 단독(Monocular) 모드 실행 파일로 변경
    orb_executable = BASE_DIR.parent / "ORB_SLAM3" / "Examples" / "Monocular" / "mono_tum"
    vocab_path = BASE_DIR.parent / "ORB_SLAM3" / "Vocabulary" / "ORBvoc.txt"
    
    # 설정 파일도 TUM 규격에 맞는 기본 파일로 변경
    yaml_path = BASE_DIR.parent / "ORB_SLAM3" / "Examples" / "Monocular" / "TUM1.yaml"
    
    # 🚨 임시 카메라 설정 파일 (나중에 스마트폰 렌즈에 맞춰 수정해야 함)
    yaml_path = BASE_DIR.parent / "ORB_SLAM3" / "Examples" / "Monocular-Inertial" / "EuRoC.yaml" 

    # 터미널 명령어 조립
    command = [
        str(orb_executable),
        str(vocab_path),
        str(yaml_path),
        str(dataset_path)
    ]

    try:
        # C++ 프로그램 실행
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # 로그 출력 (선택 사항)
        for line in process.stdout:
            print(f"[ORB-SLAM3] {line.strip()}")
            
        process.wait()
        print(f"✅ [SLAM 엔진 완료] 세션 {session_id} 처리가 끝났습니다!")
        
    except Exception as e:
        print(f"❌ SLAM 실행 중 에러 발생: {e}")

# 스트리밍 종료 시 호출될 엔드포인트
@app.post("/trigger-slam")
async def trigger_slam(session_id: str = Form(...)):
    session_dir = UPLOAD_DIR / session_id
    if not session_dir.exists():
        return {"ok": False, "message": "데이터 폴더를 찾을 수 없습니다."}
    
    # SLAM은 오래 걸리므로 서버가 멈추지 않게 별도의 쓰레드로 백그라운드 실행
    slam_thread = threading.Thread(target=run_orb_slam3, args=(session_id,))
    slam_thread.start()
    
    return {"ok": True, "message": "SLAM 엔진이 백그라운드에서 가동되었습니다."}