from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from pathlib import Path
from queue import Queue, Empty

import asyncio
import csv
import json
import os
import subprocess
import sys
import threading
import time
import zipfile


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# 경로 설정
# =========================================================
# 현재 파일 위치: 상위폴더/SLAM/main.py
BASE_DIR = Path(__file__).resolve().parent

# 상위폴더/
PROJECT_ROOT = BASE_DIR.parent

# 상위폴더/DROID-SLAM/
DROID_SLAM_DIR = Path(
    os.getenv("DROID_SLAM_DIR", PROJECT_ROOT / "DROID-SLAM")
).resolve()

# 상위폴더/DROID-SLAM/tools/imu_preintegrate.py
IMU_PREINTEGRATE_SCRIPT = Path(
    os.getenv(
        "IMU_PREINTEGRATE_SCRIPT",
        DROID_SLAM_DIR / "tools" / "imu_preintegrate.py"
    )
).resolve()

BASE_UPLOAD_DIR = Path(
    os.getenv("UPLOAD_DIR", BASE_DIR / "uploads")
).resolve()
BASE_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

IMU_PREINTEGRATE_TIMEOUT_SEC = int(
    os.getenv("IMU_PREINTEGRATE_TIMEOUT_SEC", "120")
)

# 1이면 이미지가 들어올 때마다 imu_prior.csv 갱신 요청
# 0이면 stop 때만 최종 imu_prior.csv 생성
ENABLE_IMU_PREINTEGRATION_LIVE = (
    os.getenv("ENABLE_IMU_PREINTEGRATION_LIVE", "1") == "1"
)

# 사전적분 요청 큐
# maxsize=1: 너무 많이 밀리면 오래된 요청 버리고 최신 요청만 유지
preintegration_queue = Queue(maxsize=1)

# imu_prior.csv를 동시에 여러 번 쓰지 않도록 잠금
preintegration_lock = threading.Lock()


# =========================================================
# 기본 페이지
# =========================================================
@app.get("/")
def root():
    index_path = BASE_DIR / "static" / "index.html"

    if index_path.exists():
        return FileResponse(index_path)

    return {
        "ok": True,
        "message": "server working, but static/index.html not found",
        "base_dir": str(BASE_DIR),
        "droid_slam_dir": str(DROID_SLAM_DIR),
        "imu_preintegrate_script": str(IMU_PREINTEGRATE_SCRIPT),
        "imu_script_exists": IMU_PREINTEGRATE_SCRIPT.exists(),
        "websocket": "/ws/stream",
        "sessions": "/sessions",
    }


@app.get("/routes")
def routes():
    return [route.path for route in app.routes]


# =========================================================
# 공통 유틸
# =========================================================
def sec_to_ns(timestamp_sec: float) -> int:
    return int(timestamp_sec * 1_000_000_000)


def normalize_timestamp_sec(value) -> float:
    """
    초 / 밀리초 / 나노초 timestamp를 모두 초 단위로 정규화.
    웹 Date.now()/1000, Android System.currentTimeMillis()/1000 둘 다 초 단위.
    """
    try:
        ts = float(value)
    except Exception:
        return time.time()

    if ts <= 0:
        return time.time()

    # 나노초 수준
    if ts > 1_000_000_000_000_000:
        return ts / 1_000_000_000.0

    # 밀리초 수준
    if ts > 10_000_000_000:
        return ts / 1000.0

    # 초 수준
    return ts


def safe_session_id(session_id: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    cleaned = "".join(ch for ch in str(session_id) if ch in allowed)

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
    latency_csv = session_dir / "latency.csv"

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

    if not latency_csv.exists():
        with open(latency_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "frame_id",
                "client_timestamp_sec",
                "server_saved_sec",
                "clock_latency_ms",
                "size_bytes",
            ])

    if not times_txt.exists():
        times_txt.write_text("", encoding="utf-8")

    if not calib_txt.exists():
        # DROID-SLAM calibration 형식: fx fy cx cy
        # 현재는 임시값
        calib_txt.write_text("640 480 320 240\n", encoding="utf-8")

    if not meta_json.exists():
        meta = {
            "target_slam": "DROID-SLAM",
            "communication": "single websocket",
            "websocket_endpoint": "/ws/stream",
            "image_format": "jpg_or_webp",
            "image_dir": "images",
            "frame_file": "frames.csv",
            "imu_file": "imu.csv",
            "sync_file": "synced.json",
            "imu_prior_file": "imu_prior.csv",
            "imu_preintegrate_script": str(IMU_PREINTEGRATE_SCRIPT),
            "droid_slam_dir": str(DROID_SLAM_DIR),
            "calibration_file": "calib.txt",
            "live_preintegration": ENABLE_IMU_PREINTEGRATION_LIVE,
            "protocol": {
                "start": "세션 시작",
                "frame_meta": "이미지 메타데이터 전송",
                "binary": "이미지 바이너리 전송",
                "imu": "IMU 데이터 전송",
                "stop": "마지막 데이터까지 synced.json 생성 후 imu_prior.csv 최종 생성",
            },
            "note": (
                "SLAM/main.py에서 ../DROID-SLAM/tools/imu_preintegrate.py를 실행해서 "
                "세션 폴더 안에 imu_prior.csv를 생성한다."
            ),
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


def extract_imu_values(payload: dict):
    """
    웹/앱에서 들어오는 IMU 형식이 조금 달라도 안전하게 읽기.
    """
    accel = payload.get("accel_g") or payload.get("accel") or {}
    gyro = payload.get("gyro") or {}

    ax = float(payload.get("acc_x", accel.get("x", 0.0)))
    ay = float(payload.get("acc_y", accel.get("y", 0.0)))
    az = float(payload.get("acc_z", accel.get("z", 0.0)))

    gx = float(payload.get("gyro_x", gyro.get("alpha", gyro.get("x", 0.0))))
    gy = float(payload.get("gyro_y", gyro.get("beta", gyro.get("y", 0.0))))
    gz = float(payload.get("gyro_z", gyro.get("gamma", gyro.get("z", 0.0))))

    return gx, gy, gz, ax, ay, az


# =========================================================
# 프레임-IMU 동기화 json 생성
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
                "timestamp_ns": int(float(row["timestamp_ns"])),
                "filename": row["filename"],
                "width": int(row["width"]),
                "height": int(row["height"]),
                "format": row.get("format", "jpg"),
            })

    imu_samples = []

    with open(imu_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            imu_samples.append({
                "timestamp_sec": float(row["timestamp_sec"]),
                "timestamp_ns": int(float(row["timestamp_ns"])),
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
                imu
                for imu in imu_samples
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
        avg_imu_per_frame = (
            sum(item["imu_count"] for item in synced)
            / len(synced)
        )

    return {
        "synced_path": str(synced_path),
        "frame_count": len(frames),
        "imu_count": len(imu_samples),
        "avg_imu_per_frame": avg_imu_per_frame,
    }


# =========================================================
# DROID-SLAM/tools/imu_preintegrate.py 실행
# =========================================================
def run_imu_preintegration(session_dir: Path, reason: str):
    """
    실제 실행 명령:

    python ../DROID-SLAM/tools/imu_preintegrate.py
        --session_dir uploads/세션명
        --frames uploads/세션명/frames.csv
        --imu uploads/세션명/imu.csv
        --output uploads/세션명/imu_prior.csv

    결과:
    uploads/세션명/imu_prior.csv
    """
    frames_csv = session_dir / "frames.csv"
    imu_csv = session_dir / "imu.csv"
    output_csv = session_dir / "imu_prior.csv"
    status_json = session_dir / "imu_preintegrate_status.json"
    log_txt = session_dir / "imu_preintegrate.log"

    started_at = time.time()

    def write_status(data: dict):
        status_json.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return data

    if not IMU_PREINTEGRATE_SCRIPT.exists():
        return write_status({
            "ok": False,
            "status": "script_not_found",
            "reason": reason,
            "message": f"imu_preintegrate.py not found: {IMU_PREINTEGRATE_SCRIPT}",
            "script": str(IMU_PREINTEGRATE_SCRIPT),
            "droid_slam_dir": str(DROID_SLAM_DIR),
            "output": str(output_csv),
        })

    if not frames_csv.exists():
        return write_status({
            "ok": False,
            "status": "frames_not_found",
            "reason": reason,
            "message": "frames.csv not found",
            "output": str(output_csv),
        })

    if not imu_csv.exists():
        return write_status({
            "ok": False,
            "status": "imu_not_found",
            "reason": reason,
            "message": "imu.csv not found",
            "output": str(output_csv),
        })

    command = [
        sys.executable,
        str(IMU_PREINTEGRATE_SCRIPT),
        "--session_dir", str(session_dir),
        "--frames", str(frames_csv),
        "--imu", str(imu_csv),
        "--output", str(output_csv),
    ]

    try:
        with preintegration_lock:
            with open(log_txt, "a", encoding="utf-8") as log_file:
                log_file.write("\n" + "=" * 80 + "\n")
                log_file.write("[IMU PREINTEGRATION START]\n")
                log_file.write(f"reason={reason}\n")
                log_file.write(f"started_at={started_at}\n")
                log_file.write(f"cwd={DROID_SLAM_DIR}\n")
                log_file.write("command=" + " ".join(command) + "\n")
                log_file.write("=" * 80 + "\n")
                log_file.flush()

                result = subprocess.run(
                    command,
                    cwd=str(DROID_SLAM_DIR),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=IMU_PREINTEGRATE_TIMEOUT_SEC,
                )

        finished_at = time.time()

        if result.returncode == 0:
            return write_status({
                "ok": True,
                "status": "done",
                "reason": reason,
                "message": "imu_preintegrate.py completed",
                "script": str(IMU_PREINTEGRATE_SCRIPT),
                "droid_slam_dir": str(DROID_SLAM_DIR),
                "output": str(output_csv),
                "output_exists": output_csv.exists(),
                "log": str(log_txt),
                "returncode": result.returncode,
                "started_at": started_at,
                "finished_at": finished_at,
                "elapsed_sec": round(finished_at - started_at, 3),
                "command": command,
            })

        return write_status({
            "ok": False,
            "status": "failed",
            "reason": reason,
            "message": f"imu_preintegrate.py failed with returncode={result.returncode}",
            "script": str(IMU_PREINTEGRATE_SCRIPT),
            "droid_slam_dir": str(DROID_SLAM_DIR),
            "output": str(output_csv),
            "output_exists": output_csv.exists(),
            "log": str(log_txt),
            "returncode": result.returncode,
            "started_at": started_at,
            "finished_at": finished_at,
            "elapsed_sec": round(finished_at - started_at, 3),
            "command": command,
        })

    except subprocess.TimeoutExpired:
        finished_at = time.time()

        return write_status({
            "ok": False,
            "status": "timeout",
            "reason": reason,
            "message": f"imu_preintegrate.py timeout after {IMU_PREINTEGRATE_TIMEOUT_SEC}s",
            "script": str(IMU_PREINTEGRATE_SCRIPT),
            "output": str(output_csv),
            "output_exists": output_csv.exists(),
            "log": str(log_txt),
            "started_at": started_at,
            "finished_at": finished_at,
            "elapsed_sec": round(finished_at - started_at, 3),
            "command": command,
        })

    except Exception as e:
        finished_at = time.time()

        return write_status({
            "ok": False,
            "status": "exception",
            "reason": reason,
            "message": str(e),
            "script": str(IMU_PREINTEGRATE_SCRIPT),
            "output": str(output_csv),
            "output_exists": output_csv.exists(),
            "log": str(log_txt),
            "started_at": started_at,
            "finished_at": finished_at,
            "elapsed_sec": round(finished_at - started_at, 3),
            "command": command,
        })


def clear_preintegration_queue():
    while True:
        try:
            preintegration_queue.get_nowait()
            preintegration_queue.task_done()
        except Empty:
            break
        except Exception:
            break


def request_live_preintegration(session_dir: Path, frame_id: int) -> bool:
    """
    이미지가 저장될 때마다 최신 frame/imu 기준 사전적분 요청.
    큐가 이미 차 있으면 오래된 요청은 버리고 최신 요청만 남김.
    """
    if not ENABLE_IMU_PREINTEGRATION_LIVE:
        return False

    item = {
        "session_dir": str(session_dir),
        "frame_id": frame_id,
        "reason": f"live_frame_{frame_id}",
        "requested_at": time.time(),
    }

    if preintegration_queue.full():
        try:
            preintegration_queue.get_nowait()
            preintegration_queue.task_done()
        except Empty:
            pass
        except Exception:
            pass

    try:
        preintegration_queue.put_nowait(item)
        return True
    except Exception:
        return False


def preintegration_worker():
    """
    백그라운드 worker.
    프레임이 들어올 때마다 현재까지의 frames.csv + imu.csv 기준으로 imu_prior.csv 갱신.
    """
    while True:
        item = preintegration_queue.get()

        try:
            session_dir = Path(item["session_dir"])
            reason = item.get("reason", "live")

            result = run_imu_preintegration(
                session_dir=session_dir,
                reason=reason,
            )

            print(
                f"[IMU PREINTEGRATION LIVE] "
                f"reason={reason}, "
                f"status={result.get('status')}, "
                f"ok={result.get('ok')}, "
                f"output={result.get('output')}",
                flush=True,
            )

        except Exception as e:
            print(f"[IMU PREINTEGRATION LIVE ERROR] {e}", flush=True)

        finally:
            try:
                preintegration_queue.task_done()
            except Exception:
                pass


threading.Thread(target=preintegration_worker, daemon=True).start()


# =========================================================
# WebSocket 통합 수신
# =========================================================
@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()

    print("[WEBSOCKET CONNECTED] 클라이언트 연결됨", flush=True)

    pending_frame_meta = None

    imu_received_count = 0
    frame_received_count = 0
    current_session_id = None

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
                    current_session_id = session_id
                    session_dir = get_session_dir(session_id)

                    imu_received_count = 0
                    frame_received_count = 0
                    pending_frame_meta = None

                    print(
                        f"[SESSION START] session_id={session_id}, path={session_dir}",
                        flush=True,
                    )

                    await websocket.send_text(json.dumps({
                        "ok": True,
                        "type": "started",
                        "session_id": session_id,
                        "live_preintegration": ENABLE_IMU_PREINTEGRATION_LIVE,
                        "imu_preintegrate_script": str(IMU_PREINTEGRATE_SCRIPT),
                        "imu_script_exists": IMU_PREINTEGRATE_SCRIPT.exists(),
                    }, ensure_ascii=False))

                # ---------------------------------------------
                # 프레임 메타데이터 수신
                # ---------------------------------------------
                elif msg_type == "frame_meta":
                    session_id = payload.get("session_id") or current_session_id

                    if not session_id:
                        await websocket.send_text(json.dumps({
                            "ok": False,
                            "type": "error",
                            "message": "session_id missing in frame_meta",
                        }, ensure_ascii=False))
                        continue

                    payload["session_id"] = safe_session_id(session_id)
                    pending_frame_meta = payload

                # ---------------------------------------------
                # IMU 수신
                # ---------------------------------------------
                elif msg_type == "imu":
                    session_id = payload.get("session_id") or current_session_id

                    if not session_id:
                        await websocket.send_text(json.dumps({
                            "ok": False,
                            "type": "error",
                            "message": "session_id missing in imu",
                        }, ensure_ascii=False))
                        continue

                    session_id = safe_session_id(session_id)
                    session_dir = get_session_dir(session_id)

                    timestamp_sec = normalize_timestamp_sec(
                        payload.get("timestamp", 0.0)
                    )
                    timestamp_ns = sec_to_ns(timestamp_sec)

                    gx, gy, gz, ax, ay, az = extract_imu_values(payload)

                    with open(
                        session_dir / "imu.csv",
                        "a",
                        newline="",
                        encoding="utf-8",
                    ) as f:
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
                        print(
                            f"[IMU RECEIVED] "
                            f"count={imu_received_count}, "
                            f"time={timestamp_sec:.6f}, "
                            f"gyro=({gx:.4f}, {gy:.4f}, {gz:.4f}), "
                            f"accel=({ax:.4f}, {ay:.4f}, {az:.4f})",
                            flush=True,
                        )

                        await websocket.send_text(json.dumps({
                            "ok": True,
                            "type": "imu_saved",
                            "imu_received_count": imu_received_count,
                            "timestamp_ns": timestamp_ns,
                        }, ensure_ascii=False))

                # ---------------------------------------------
                # 수집 종료
                # stop이 오면 바로 끊지 않고:
                # 1. 현재까지 저장된 마지막 frame/imu 기준 synced.json 생성
                # 2. live queue 비움
                # 3. 최종 imu_preintegrate.py 실행 완료
                # 4. stopped 응답
                # ---------------------------------------------
                elif msg_type == "stop":
                    session_id = payload.get("session_id") or current_session_id

                    if not session_id:
                        await websocket.send_text(json.dumps({
                            "ok": False,
                            "type": "error",
                            "message": "session_id missing in stop",
                        }, ensure_ascii=False))
                        continue

                    session_id = safe_session_id(session_id)
                    session_dir = get_session_dir(session_id)

                    print(
                        f"[SESSION STOP REQUEST] session_id={session_id}",
                        flush=True,
                    )

                    await websocket.send_text(json.dumps({
                        "ok": True,
                        "type": "preintegration_started",
                        "message": "마지막 frame/imu 기준 synced.json 생성 및 최종 imu_prior.csv 생성 시작",
                        "session_id": session_id,
                    }, ensure_ascii=False))

                    # 현재까지 저장된 마지막 frame/imu 기준 synced.json 생성
                    sync_result = build_synced_json(session_dir)

                    # stop 최종 계산 전에 밀린 live 요청 제거
                    clear_preintegration_queue()

                    # 최종 사전적분은 반드시 끝날 때까지 기다림
                    final_preintegration_result = await asyncio.to_thread(
                        run_imu_preintegration,
                        session_dir,
                        "stop_final",
                    )

                    droid_command = (
                        f"python demo.py "
                        f"--imagedir={session_dir / 'images'} "
                        f"--calib={session_dir / 'calib.txt'} "
                        f"--disable_vis "
                        f"--reconstruction_path={session_dir / 'reconstruction.pth'}"
                    )

                    print(
                        f"[SESSION STOP DONE] "
                        f"session_id={session_id}, "
                        f"frames={sync_result['frame_count']}, "
                        f"imu={sync_result['imu_count']}, "
                        f"avg_imu_per_frame={sync_result['avg_imu_per_frame']:.2f}, "
                        f"preintegration={final_preintegration_result.get('status')}",
                        flush=True,
                    )

                    await websocket.send_text(json.dumps({
                        "ok": True,
                        "type": "stopped",
                        "message": "DROID-SLAM용 데이터 생성 및 최종 imu_prior.csv 생성 완료",
                        "session_id": session_id,
                        "session_dir": str(session_dir),
                        "droid_ready": True,
                        "droid_command": droid_command,
                        "preintegration": final_preintegration_result,
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

                session_id = pending_frame_meta.get("session_id") or current_session_id

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

                image_format = str(
                    pending_frame_meta.get("format", "jpg")
                ).lower()

                if image_format in ["jpg", "jpeg"]:
                    ext = "jpg"
                    save_format = "jpg"
                elif image_format == "webp":
                    ext = "webp"
                    save_format = "webp"
                else:
                    ext = "jpg"
                    save_format = "jpg"

                filename = f"{frame_id:06d}.{ext}"
                image_path = images_dir / filename

                with open(image_path, "wb") as f:
                    f.write(image_bytes)

                timestamp_sec = normalize_timestamp_sec(
                    pending_frame_meta.get("timestamp", 0.0)
                )
                timestamp_ns = sec_to_ns(timestamp_sec)

                width = int(pending_frame_meta.get("width", 640))
                height = int(pending_frame_meta.get("height", 480))
                size_bytes = int(
                    pending_frame_meta.get("size_bytes", len(image_bytes))
                )

                server_saved_sec = time.time()
                clock_latency_ms = (server_saved_sec - timestamp_sec) * 1000.0

                client_send_perf_ms = pending_frame_meta.get("client_send_perf_ms")
                capture_start_perf_ms = pending_frame_meta.get("capture_start_perf_ms")

                with open(
                    session_dir / "frames.csv",
                    "a",
                    newline="",
                    encoding="utf-8",
                ) as f:
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

                with open(
                    session_dir / "times.txt",
                    "a",
                    encoding="utf-8",
                ) as f:
                    f.write(f"{timestamp_sec:.9f}\n")

                with open(
                    session_dir / "latency.csv",
                    "a",
                    newline="",
                    encoding="utf-8",
                ) as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        frame_id,
                        f"{timestamp_sec:.9f}",
                        f"{server_saved_sec:.9f}",
                        f"{clock_latency_ms:.3f}",
                        size_bytes,
                    ])

                frame_received_count += 1

                # 이미지가 저장될 때마다 바로 사전적분 요청
                live_preintegration_queued = request_live_preintegration(
                    session_dir=session_dir,
                    frame_id=frame_id,
                )

                print(
                    f"[FRAME RECEIVED] "
                    f"count={frame_received_count}, "
                    f"frame_id={frame_id}, "
                    f"file={filename}, "
                    f"bytes={len(image_bytes)}, "
                    f"size={width}x{height}, "
                    f"format={save_format}, "
                    f"time={timestamp_sec:.6f}, "
                    f"preintegration_queued={live_preintegration_queued}",
                    flush=True,
                )

                await websocket.send_text(json.dumps({
                    "ok": True,
                    "type": "frame_saved",
                    "session_id": session_id,
                    "frame_received_count": frame_received_count,
                    "frame_id": frame_id,
                    "filename": filename,
                    "format": save_format,
                    "timestamp_ns": timestamp_ns,
                    "size_bytes": size_bytes,
                    "clock_latency_ms": round(clock_latency_ms, 2),
                    "client_send_perf_ms": client_send_perf_ms,
                    "capture_start_perf_ms": capture_start_perf_ms,
                    "live_preintegration_queued": live_preintegration_queued,
                }, ensure_ascii=False))

                pending_frame_meta = None

    except WebSocketDisconnect:
        print("[WEBSOCKET DISCONNECTED] 클라이언트 연결 종료", flush=True)

    except Exception as e:
        print(f"[WEBSOCKET ERROR] {e}", flush=True)

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
    imu_prior_path = session_dir / "imu_prior.csv"
    status_path = session_dir / "imu_preintegrate_status.json"

    frame_count = 0
    imu_count = 0
    synced_count = 0
    imu_prior_count = 0

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

    if imu_prior_path.exists():
        with open(imu_prior_path, "r", newline="", encoding="utf-8") as f:
            imu_prior_count = max(0, len(list(csv.reader(f))) - 1)

    preintegration_status = None

    if status_path.exists():
        try:
            preintegration_status = json.loads(
                status_path.read_text(encoding="utf-8")
            )
        except Exception:
            preintegration_status = {
                "ok": False,
                "status": "status_json_read_failed",
            }

    return {
        "ok": True,
        "session_id": session_id,
        "session_dir": str(session_dir),
        "frame_count": frame_count,
        "imu_count": imu_count,
        "synced_count": synced_count,
        "imu_prior_count": imu_prior_count,
        "imu_prior_exists": imu_prior_path.exists(),
        "preintegration_status": preintegration_status,
        "imu_preintegrate_script": str(IMU_PREINTEGRATE_SCRIPT),
        "imu_script_exists": IMU_PREINTEGRATE_SCRIPT.exists(),
    }


# =========================================================
# 저장 세션 목록 확인용
# =========================================================
@app.get("/sessions")
def list_sessions():
    sessions = []

    if not BASE_UPLOAD_DIR.exists():
        return {
            "ok": True,
            "sessions": [],
        }

    for session_dir in sorted(BASE_UPLOAD_DIR.iterdir(), reverse=True):
        if not session_dir.is_dir():
            continue

        frames_path = session_dir / "frames.csv"
        imu_path = session_dir / "imu.csv"
        synced_path = session_dir / "synced.json"
        imu_prior_path = session_dir / "imu_prior.csv"
        status_path = session_dir / "imu_preintegrate_status.json"

        frame_count = 0
        imu_count = 0
        synced_count = 0
        imu_prior_count = 0

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

        if imu_prior_path.exists():
            with open(imu_prior_path, "r", newline="", encoding="utf-8") as f:
                imu_prior_count = max(0, len(list(csv.reader(f))) - 1)

        preintegration_status = "not_run"

        if status_path.exists():
            try:
                status_json = json.loads(status_path.read_text(encoding="utf-8"))
                preintegration_status = status_json.get("status", "unknown")
            except Exception:
                preintegration_status = "status_read_failed"

        sessions.append({
            "session_id": session_dir.name,
            "frame_count": frame_count,
            "imu_count": imu_count,
            "synced_count": synced_count,
            "imu_prior_count": imu_prior_count,
            "imu_prior_exists": imu_prior_path.exists(),
            "preintegration_status": preintegration_status,
            "summary_url": f"/session/{session_dir.name}/summary",
            "download_url": f"/session/{session_dir.name}/download",
            "preintegrate_url": f"/session/{session_dir.name}/preintegrate",
        })

    return {
        "ok": True,
        "sessions": sessions,
    }


# =========================================================
# 저장 세션 ZIP 다운로드
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
                "message": f"session not found: {session_id}",
            },
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
        filename=f"{session_id}.zip",
    )


# =========================================================
# 수동 IMU 사전적분 실행 API
# =========================================================
@app.post("/session/{session_id}/preintegrate")
async def preintegrate_api(session_id: str):
    session_id = safe_session_id(session_id)
    session_dir = BASE_UPLOAD_DIR / session_id

    if not session_dir.exists() or not session_dir.is_dir():
        return JSONResponse(
            status_code=404,
            content={
                "ok": False,
                "message": f"session not found: {session_id}",
            },
        )

    sync_result = build_synced_json(session_dir)

    clear_preintegration_queue()

    preintegration_result = await asyncio.to_thread(
        run_imu_preintegration,
        session_dir,
        "manual_api",
    )

    return {
        "ok": True,
        "session_id": session_id,
        "session_dir": str(session_dir),
        "preintegration": preintegration_result,
        **sync_result,
    }


# =========================================================
# 기존 HTTP trigger 비활성화 안내
# =========================================================
@app.post("/trigger-slam")
async def trigger_slam_legacy():
    return JSONResponse(
        status_code=400,
        content={
            "ok": False,
            "message": (
                "이 버전에서는 /trigger-slam을 사용하지 않습니다. "
                "/ws/stream의 stop 메시지 또는 /session/{session_id}/preintegrate를 사용하세요."
            ),
        },
    )
