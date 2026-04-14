from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import os
import glob
import torch
import numpy as np
import gc
import cv2

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


# ----------------------
# 세션 폴더 생성
# ----------------------
def create_session_dirs():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    session_dir = BASE_UPLOAD_DIR / f"input_{timestamp}"
    images_dir = session_dir / "images"
    videos_dir = session_dir / "videos"
    images_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    return session_dir, images_dir, videos_dir


def save_upload_file(upload_file: UploadFile, save_dir: Path):
    file_path = save_dir / upload_file.filename
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return str(file_path)


# ----------------------
# FastAPI 루트
# ----------------------
@app.get("/")
def root():
    return {"message": "server working"}


# ----------------------
# 업로드만 하는 기존 엔드포인트
# ----------------------
@app.post("/upload")
async def upload_files(
    images: Optional[List[UploadFile]] = File(default=None),
    video: Optional[UploadFile] = File(default=None),
    user_id: Optional[str] = Form(default="anonymous"),
):
    if not images and not video:
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": "업로드된 파일이 없습니다."},
        )

    session_dir, images_dir, videos_dir = create_session_dirs()

    saved_images = []
    saved_video = None

    if images:
        for image in images:
            if image.filename:
                path = save_upload_file(image, images_dir)
                saved_images.append(path)

    if video and video.filename:
        saved_video = save_upload_file(video, videos_dir)

    return {
        "success": True,
        "user_id": user_id,
        "session_dir": str(session_dir),
        "image_count": len(saved_images),
        "image_paths": saved_images,
        "video_path": saved_video,
    }


# ----------------------
# 모델 좌표 계산용 predict 엔드포인트
# ----------------------
@app.post("/predict")
async def predict(
    images: Optional[List[UploadFile]] = File(default=None),
    video: Optional[UploadFile] = File(default=None),
    user_id: Optional[str] = Form(default="anonymous"),
):
    """
    사진/영상 업로드 후 30fps 프레임 추출 및 파일 경로 JSON 반환
    """
    if not images and not video:
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": "업로드된 파일이 없습니다."},
        )

    session_dir, images_dir, videos_dir = create_session_dirs()
    saved_images = []
    saved_video_path = None

    # 1. 사진(이미지) 업로드 시 저장
    if images:
        for img in images:
            if img.filename:
                path = save_upload_file(img, images_dir)
                saved_images.append(path)

    # 2. 동영상 업로드 시 30fps 프레임 추출 및 times.txt 생성
    if video and video.filename:
        saved_video_path = save_upload_file(video, videos_dir)
        
        vs = cv2.VideoCapture(saved_video_path)
        
        # 원본 영상의 fps 확인
        original_fps = vs.get(cv2.CAP_PROP_FPS) or 30.0
        
        # 목표 fps를 30으로 설정 (60fps 영상이면 2프레임마다 1장씩, 30fps면 매 프레임 추출)
        target_fps = 30.0
        frame_skip_interval = max(int(original_fps / target_fps), 1)
        
        times_file_path = session_dir / "times.txt"
        time_interval = 1.0 / target_fps
        
        count = 0
        frame_idx = 0
        
        with open(times_file_path, "w") as f:
            while True:
                ret, frame = vs.read()
                if not ret:
                    break
                
                # 계산된 간격(30fps)에 맞춰 프레임 저장
                if count % frame_skip_interval == 0:
                    frame_filename = f"frame_{frame_idx:06d}.png"
                    frame_path = images_dir / frame_filename
                    cv2.imwrite(str(frame_path), frame)
                    saved_images.append(str(frame_path))
                    
                    # SLAM용 타임스탬프 기록
                    f.write(f"{frame_idx * time_interval:.6f}\n")
                    frame_idx += 1
                
                count += 1
                
        vs.release()

    if len(saved_images) == 0:
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": "처리할 이미지가 없습니다."},
        )

    # ----------------------
    # 3. 첨부한 이미지와 동일한 구조의 JSON 응답 반환
    # ----------------------
    return {
        "success": True,
        "message": f"이미지가 성공적으로 업로드/추출되었습니다! (총 {len(saved_images)}장 저장됨)",
        "user_id": user_id,
        "session_dir": str(session_dir),
        "image_count": len(saved_images),
        "image_paths": saved_images,
        "video_path": saved_video_path
    }