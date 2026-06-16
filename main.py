from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from pathlib import Path
import csv
import json
import zipfile


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_UPLOAD_DIR = Path("uploads")
BASE_UPLOAD_DIR.mkdir(exist_ok=True)


# =========================================================
# 기본 페이지
# =========================================================
@app.get("/")
def root():
    index_path = Path("./static/index.html")

    if index_path.exists():
        return FileResponse(index_path)

    return {"message": "server working, but static/index.html not found"}


# =========================================================
# 라우트 확인용
# =========================================================
@app.get("/routes")
def routes():
    return [route.path for route in app.routes]


# =========================================================
# 공통 유틸
# =========================================================
def sec_to_ns(timestamp_sec: float) -> int:
    return int(timestamp_sec * 1_000_000_000)


def ms_to_sec(timestamp_ms: float) -> float:
    return timestamp_ms / 1000.0


def safe_session_id(session_id: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    cleaned = "".join(ch for ch in session_id if ch in allowed)

    if cleaned == "":
        cleaned = "session_default"

    return cleaned


def get_session_dir(session_id: str) -> Path:
    session_id = safe_session_id(session_id)

    session_dir = BASE_UPLOAD_DIR / session_id
    images_dir = session_dir / "images"

    session_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    init_session_files(session_dir)

    return session_dir


def init_session_files(session_dir: Path):
    frames_csv = session_dir / "frames.csv"
    imu_csv = session_dir / "imu.csv"
    times_txt = session_dir / "times.txt"
    calib_txt = session_dir / "calib.txt"
    meta_json = session_dir / "meta.json"

    if not frames_csv.exists():
        with open(frames_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "frame_id",
                "timestamp_sec",
                "timestamp_ns",
                "filename",
                "width",
                "height",
                "format",
            ])

    if not imu_csv.exists():
        with open(imu_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp_sec",
                "timestamp_ns",
                "gx",
                "gy",
                "gz",
                "ax",
                "ay",
                "az",
            ])

    if not times_txt.exists():
        times_txt.write_text("", encoding="utf-8")

    if not calib_txt.exists():
        # DROID-SLAM calibration 형식: fx fy cx cy
        # 현재는 임시값
        calib_txt.write_text("517.3 516.5 318.6 255.3\n", encoding="utf-8")

    if not meta_json.exists():
        meta = {
            "target_slam": "DROID-SLAM",
            "communication": "single websocket",
            "websocket_endpoint": "/ws/stream",
            "image_format": "webp_or_jpg",
            "image_dir": "images",
            "frame_file": "frames.csv",
            "imu_file": "imu.csv",
            "sync_file": "synced.json",
            "calibration_file": "calib.txt",
            "protocol": {
                "start": "세션 시작",
                "frame_meta": "이미지 메타데이터 전송",
                "binary": "이미지 바이너리 전송",
                "imu": "IMU 데이터 전송",
                "stop": "수집 종료 및 synced.json 생성",
            },
            "frame_format": "timestamp_ms,width,height,format + binary image bytes",
            "imu_format": "timestamp_sec,timestamp_ns,gx,gy,gz,ax,ay,az",
            "gyro_unit": "rad/s",
            "accel_unit": "m/s^2",
            "note": "이미지와 IMU를 하나의 WebSocket(/ws/stream)으로 수신한다.",
        }

        meta_json.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def get_next_frame_id(session_dir: Path) -> int:
    frames_csv = session_dir / "frames.csv"

    if not frames_csv.exists():
        return 0

    with open(frames_csv, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    return max(0, len(rows) - 1)


# =========================================================
# 프레임-IMU 동기화
# =========================================================
def build_synced_json(session_dir: Path):
    frames_path = session_dir / "frames.csv"
    imu_path = session_dir / "imu.csv"
    synced_path = session_dir / "synced.json"

    if not frames_path.exists():
        raise FileNotFoundError("frames.csv not found")

    if not imu_path.exists():
        raise FileNotFoundError("imu.csv not found")

    frames = []
    with open(frames_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            frames.append({
                "frame_id": int(row["frame_id"]),
                "timestamp_sec": float(row["timestamp_sec"]),
                "timestamp_ns": int(row["timestamp_ns"]),
                "filename": row["filename"],
                "width": int(row["width"]),
                "height": int(row["height"]),
                "format": row.get("format", "webp"),
            })

    imu_samples = []
    with open(imu_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            imu_samples.append({
                "timestamp_sec": float(row["timestamp_sec"]),
                "timestamp_ns": int(row["timestamp_ns"]),
                "gx": float(row["gx"]),
                "gy": float(row["gy"]),
                "gz": float(row["gz"]),
                "ax": float(row["ax"]),
                "ay": float(row["ay"]),
                "az": float(row["az"]),
            })

    frames.sort(key=lambda x: x["timestamp_ns"])
    imu_samples.sort(key=lambda x: x["timestamp_ns"])

    synced = []

    for i, frame in enumerate(frames):
        curr_ts = frame["timestamp_ns"]

        if i == 0:
            prev_ts = None
            imu_window = []
        else:
            prev_ts = frames[i - 1]["timestamp_ns"]
            imu_window = [
                imu for imu in imu_samples
                if prev_ts < imu["timestamp_ns"] <= curr_ts
            ]

        synced.append({
            "frame_id": frame["frame_id"],
            "timestamp_sec": frame["timestamp_sec"],
            "timestamp_ns": frame["timestamp_ns"],
            "image": f"images/{frame['filename']}",
            "width": frame["width"],
            "height": frame["height"],
            "format": frame["format"],
            "imu_range": {
                "start_timestamp_ns": prev_ts,
                "end_timestamp_ns": curr_ts,
            },
            "imu_count": len(imu_window),
            "imu": imu_window,
        })

    synced_path.write_text(
        json.dumps(synced, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    avg_imu_per_frame = 0.0

    if len(synced) > 0:
        avg_imu_per_frame = sum(item["imu_count"] for item in synced) / len(synced)

    return {
        "synced_path": str(synced_path),
        "frame_count": len(frames),
        "imu_count": len(imu_samples),
        "avg_imu_per_frame": avg_imu_per_frame,
    }


# =========================================================
# WebSocket 통합 수신
# =========================================================
@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()

    pending_frame_meta = None

    imu_received_count = 0
    frame_received_count = 0

    try:
        while True:
            message = await websocket.receive()

            # -------------------------------------------------
            # JSON 메시지 수신
            # -------------------------------------------------
            if message.get("text") is not None:
                try:
                    payload = json.loads(message["text"])
                except json.JSONDecodeError:
                    await websocket.send_text(json.dumps({
                        "ok": False,
                        "type": "error",
                        "message": "invalid json message",
                    }, ensure_ascii=False))
                    continue

                msg_type = payload.get("type")

                # ---------------------------------------------
                # 세션 시작
                # ---------------------------------------------
                if msg_type == "start":
                    session_id = payload.get("session_id")

                    if not session_id:
                        await websocket.send_text(json.dumps({
                            "ok": False,
                            "type": "error",
                            "message": "session_id missing",
                        }, ensure_ascii=False))
                        continue

                    session_id = safe_session_id(session_id)
                    get_session_dir(session_id)

                    imu_received_count = 0
                    frame_received_count = 0

                    await websocket.send_text(json.dumps({
                        "ok": True,
                        "type": "started",
                        "session_id": session_id,
                    }, ensure_ascii=False))

                # ---------------------------------------------
                # 프레임 메타데이터 수신
                # ---------------------------------------------
                elif msg_type == "frame_meta":
                    session_id = payload.get("session_id")

                    if not session_id:
                        await websocket.send_text(json.dumps({
                            "ok": False,
                            "type": "error",
                            "message": "session_id missing in frame_meta",
                        }, ensure_ascii=False))
                        continue

                    pending_frame_meta = payload

                # ---------------------------------------------
                # IMU 수신
                # Android SensorManager:
                # accelerometer: m/s^2
                # gyroscope: rad/s
                # ---------------------------------------------
                elif msg_type == "imu":
                    session_id = payload.get("session_id")

                    if not session_id:
                        await websocket.send_text(json.dumps({
                            "ok": False,
                            "type": "error",
                            "message": "session_id missing in imu",
                        }, ensure_ascii=False))
                        continue

                    session_id = safe_session_id(session_id)
                    session_dir = get_session_dir(session_id)

                    timestamp_ms = float(payload.get("timestamp", 0.0))
                    timestamp_sec = ms_to_sec(timestamp_ms)
                    timestamp_ns = sec_to_ns(timestamp_sec)

                    accel = payload.get("accel_g", {})
                    gyro = payload.get("gyro", {})

                    gx = float(gyro.get("alpha", 0.0))
                    gy = float(gyro.get("beta", 0.0))
                    gz = float(gyro.get("gamma", 0.0))

                    ax = float(accel.get("x", 0.0))
                    ay = float(accel.get("y", 0.0))
                    az = float(accel.get("z", 0.0))

                    with open(session_dir / "imu.csv", "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            f"{timestamp_sec:.9f}",
                            timestamp_ns,
                            gx,
                            gy,
                            gz,
                            ax,
                            ay,
                            az,
                        ])

                    imu_received_count += 1

                    if imu_received_count % 200 == 0:
                        await websocket.send_text(json.dumps({
                            "ok": True,
                            "type": "imu_saved",
                            "imu_received_count": imu_received_count,
                            "timestamp_ns": timestamp_ns,
                        }, ensure_ascii=False))

                # ---------------------------------------------
                # 수집 종료 및 synced.json 생성
                # ---------------------------------------------
                elif msg_type == "stop":
                    session_id = payload.get("session_id")

                    if not session_id:
                        await websocket.send_text(json.dumps({
                            "ok": False,
                            "type": "error",
                            "message": "session_id missing in stop",
                        }, ensure_ascii=False))
                        continue

                    session_id = safe_session_id(session_id)
                    session_dir = get_session_dir(session_id)

                    sync_result = build_synced_json(session_dir)

                    droid_command = (
                        f"python demo.py "
                        f"--imagedir={session_dir / 'images'} "
                        f"--calib={session_dir / 'calib.txt'} "
                        f"--disable_vis "
                        f"--reconstruction_path={session_dir / 'reconstruction.pth'}"
                    )

                    await websocket.send_text(json.dumps({
                        "ok": True,
                        "type": "stopped",
                        "message": "DROID-SLAM용 데이터 생성 완료",
                        "session_id": session_id,
                        "session_dir": str(session_dir),
                        "droid_ready": True,
                        "droid_command": droid_command,
                        **sync_result,
                    }, ensure_ascii=False))

                else:
                    await websocket.send_text(json.dumps({
                        "ok": False,
                        "type": "error",
                        "message": f"unknown message type: {msg_type}",
                    }, ensure_ascii=False))

            # -------------------------------------------------
            # 이미지 binary bytes 수신
            # 반드시 직전에 frame_meta가 와야 함
            # -------------------------------------------------
            elif message.get("bytes") is not None:
                image_bytes = message["bytes"]

                if pending_frame_meta is None:
                    await websocket.send_text(json.dumps({
                        "ok": False,
                        "type": "error",
                        "message": "image binary received but frame_meta missing",
                    }, ensure_ascii=False))
                    continue

                session_id = pending_frame_meta.get("session_id")

                if not session_id:
                    await websocket.send_text(json.dumps({
                        "ok": False,
                        "type": "error",
                        "message": "session_id missing in pending frame_meta",
                    }, ensure_ascii=False))
                    pending_frame_meta = None
                    continue

                session_id = safe_session_id(session_id)
                session_dir = get_session_dir(session_id)
                images_dir = session_dir / "images"

                frame_id = get_next_frame_id(session_dir)

                image_format = str(pending_frame_meta.get("format", "webp")).lower()

                if image_format in ["jpg", "jpeg"]:
                    ext = "jpg"
                    save_format = "jpg"
                else:
                    ext = "webp"
                    save_format = "webp"

                filename = f"{frame_id:06d}.{ext}"
                image_path = images_dir / filename

                with open(image_path, "wb") as f:
                    f.write(image_bytes)

                timestamp_ms = float(pending_frame_meta.get("timestamp", 0.0))
                timestamp_sec = ms_to_sec(timestamp_ms)
                timestamp_ns = sec_to_ns(timestamp_sec)

                width = int(pending_frame_meta.get("width", 640))
                height = int(pending_frame_meta.get("height", 480))

                with open(session_dir / "frames.csv", "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        frame_id,
                        f"{timestamp_sec:.9f}",
                        timestamp_ns,
                        filename,
                        width,
                        height,
                        save_format,
                    ])

                with open(session_dir / "times.txt", "a", encoding="utf-8") as f:
                    f.write(f"{timestamp_sec:.9f}\n")

                frame_received_count += 1

                if frame_received_count % 10 == 0:
                    await websocket.send_text(json.dumps({
                        "ok": True,
                        "type": "frame_saved",
                        "frame_received_count": frame_received_count,
                        "frame_id": frame_id,
                        "filename": filename,
                        "format": save_format,
                        "timestamp_ns": timestamp_ns,
                    }, ensure_ascii=False))

                pending_frame_meta = None

    except WebSocketDisconnect:
        print("stream websocket disconnected")

    except Exception as e:
        print(f"stream websocket error: {e}")

        try:
            await websocket.send_text(json.dumps({
                "ok": False,
                "type": "error",
                "message": str(e),
            }, ensure_ascii=False))
        except Exception:
            pass


# =========================================================
# 세션 확인용
# =========================================================
@app.get("/session/{session_id}/summary")
def session_summary(session_id: str):
    session_id = safe_session_id(session_id)
    session_dir = get_session_dir(session_id)

    frames_path = session_dir / "frames.csv"
    imu_path = session_dir / "imu.csv"
    synced_path = session_dir / "synced.json"

    frame_count = 0
    imu_count = 0
    synced_count = 0

    if frames_path.exists():
        with open(frames_path, "r", newline="", encoding="utf-8") as f:
            frame_count = max(0, len(list(csv.reader(f))) - 1)

    if imu_path.exists():
        with open(imu_path, "r", newline="", encoding="utf-8") as f:
            imu_count = max(0, len(list(csv.reader(f))) - 1)

    if synced_path.exists():
        synced = json.loads(synced_path.read_text(encoding="utf-8"))
        synced_count = len(synced)

    return {
        "ok": True,
        "session_id": session_id,
        "session_dir": str(session_dir),
        "frame_count": frame_count,
        "imu_count": imu_count,
        "synced_count": synced_count,
    }


# =========================================================
# Render 저장 세션 목록 확인용
# =========================================================
@app.get("/sessions")
def list_sessions():
    sessions = []

    if not BASE_UPLOAD_DIR.exists():
        return {
            "ok": True,
            "sessions": []
        }

    for session_dir in sorted(BASE_UPLOAD_DIR.iterdir(), reverse=True):
        if not session_dir.is_dir():
            continue

        frames_path = session_dir / "frames.csv"
        imu_path = session_dir / "imu.csv"
        synced_path = session_dir / "synced.json"

        frame_count = 0
        imu_count = 0
        synced_count = 0

        if frames_path.exists():
            with open(frames_path, "r", newline="", encoding="utf-8") as f:
                frame_count = max(0, len(list(csv.reader(f))) - 1)

        if imu_path.exists():
            with open(imu_path, "r", newline="", encoding="utf-8") as f:
                imu_count = max(0, len(list(csv.reader(f))) - 1)

        if synced_path.exists():
            try:
                synced = json.loads(synced_path.read_text(encoding="utf-8"))
                synced_count = len(synced)
            except Exception:
                synced_count = 0

        sessions.append({
            "session_id": session_dir.name,
            "frame_count": frame_count,
            "imu_count": imu_count,
            "synced_count": synced_count,
            "summary_url": f"/session/{session_dir.name}/summary",
            "download_url": f"/session/{session_dir.name}/download",
        })

    return {
        "ok": True,
        "sessions": sessions
    }


# =========================================================
# Render 저장 세션 ZIP 다운로드
# =========================================================
@app.get("/session/{session_id}/download")
def download_session(session_id: str):
    session_id = safe_session_id(session_id)
    session_dir = BASE_UPLOAD_DIR / session_id

    if not session_dir.exists() or not session_dir.is_dir():
        return JSONResponse(
            status_code=404,
            content={
                "ok": False,
                "message": f"session not found: {session_id}"
            }
        )

    zip_path = session_dir / f"{session_id}.zip"

    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for file_path in session_dir.rglob("*"):
            if file_path == zip_path:
                continue

            if file_path.is_file():
                arcname = file_path.relative_to(session_dir.parent)
                zip_file.write(file_path, arcname)

    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"{session_id}.zip"
    )


# =========================================================
# 기존 HTTP trigger 비활성화 안내
# =========================================================
@app.post("/trigger-slam")
async def trigger_slam_legacy():
    return JSONResponse(
        status_code=400,
        content={
            "ok": False,
            "message": "이 버전에서는 /trigger-slam을 사용하지 않습니다. /ws/stream의 stop 메시지를 사용하세요.",
        },
    )
