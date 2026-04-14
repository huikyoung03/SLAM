from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import json
import os
import time

app = FastAPI(title="Mobile IMU + Frame Demo")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# 세션별 마지막 IMU 타임스탬프 저장
last_imu_timestamp = {}
imu_counts = {}
frame_counts = {}

@app.get("/", response_class=HTMLResponse)
async def index():
    index_file = STATIC_DIR / "index.html"
    return index_file.read_text(encoding="utf-8")

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.websocket("/ws/mobile")
async def websocket_mobile(websocket: WebSocket):
    await websocket.accept()
    session_id = str(id(websocket))
    imu_counts[session_id] = 0

    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)

            if data.get("type") == "imu":
                imu_counts[session_id] += 1
                last_imu_timestamp[session_id] = data.get("timestamp")

                # 실제 프로젝트에서는 여기서 버퍼에 저장
                print(
                    f"[IMU] session={session_id} count={imu_counts[session_id]} "
                    f"ts={data.get('timestamp')} "
                    f"accel={data.get('accel')} gyro={data.get('gyro')}"
                )

                await websocket.send_text(json.dumps({
                    "type": "ack",
                    "session_id": session_id,
                    "imu_count": imu_counts[session_id]
                }))

            elif data.get("type") == "hello":
                await websocket.send_text(json.dumps({
                    "type": "hello_ack",
                    "message": "websocket connected",
                    "session_id": session_id
                }))

    except WebSocketDisconnect:
        print(f"WebSocket disconnected: {session_id}")

@app.post("/frame")
async def upload_frame(
    session_id: str = Form(...),
    timestamp: float = Form(...),
    frame: UploadFile = File(...)
):
    frame_counts[session_id] = frame_counts.get(session_id, 0) + 1

    filename = f"{session_id}_{int(timestamp)}_{frame_counts[session_id]}.jpg"
    save_path = UPLOAD_DIR / filename

    content = await frame.read()
    with open(save_path, "wb") as f:
        f.write(content)

    print(
        f"[FRAME] session={session_id} count={frame_counts[session_id]} "
        f"ts={timestamp} saved={save_path.name}"
    )

    return JSONResponse({
        "ok": True,
        "session_id": session_id,
        "frame_count": frame_counts[session_id],
        "timestamp": timestamp,
        "saved": save_path.name
    })