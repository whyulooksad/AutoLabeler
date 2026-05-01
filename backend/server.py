import argparse
import gc
import io
import logging
import os
import shutil
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TORCH_CACHE_DIR = PROJECT_ROOT / "checkpoints" / "torch_cache"
TORCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TORCH_HOME", str(TORCH_CACHE_DIR))

import cv2
import numpy as np
import requests
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from PIL import Image
from tqdm import tqdm

os.chdir(PROJECT_ROOT)

from backend.sampled_dataset_generator import get_sampled_generator
from backend.tools.painter import mask_painter
from backend.track_anything import TrackingAnything
from backend.video_processor import get_video_processor

logger = logging.getLogger("autolabeler")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SAM_CHECKPOINTS = {
    "vit_h": ("sam_vit_h_4b8939.pth", "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"),
    "vit_l": ("sam_vit_l_0b3195.pth", "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"),
    "vit_b": ("sam_vit_b_01ec64.pth", "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"),
}
XMEM_CHECKPOINT = ("XMem-s012.pth", "https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem-s012.pth")


class ClickRequest(BaseModel):
    x: int
    y: int
    label: Literal["positive", "negative"] = "positive"


class SelectFrameRequest(BaseModel):
    frame_index: int


class TrackRequest(BaseModel):
    mask_names: list[str] = []
    end_frame: int | None = None
    xmem_size: int | None = None


class YoloRequest(BaseModel):
    mask_names: list[str] = []
    end_frame: int | None = None


class AddMaskRequest(BaseModel):
    class_name: str


class UpdateMaskRequest(BaseModel):
    class_name: str


@dataclass
class SessionState:
    id: str
    video_name: str
    original_video_path: str
    processed_video_path: str
    transcode_info: dict | None
    origin_images: list[np.ndarray]
    painted_images: list[np.ndarray]
    masks: list[np.ndarray]
    logits: list[object]
    fps: float
    select_frame_number: int = 0
    track_end_number: int | None = None
    click_points: list[list[int]] = field(default_factory=list)
    click_labels: list[int] = field(default_factory=list)
    mask_names: list[str] = field(default_factory=list)
    class_names: list[str] = field(default_factory=list)
    mask_class_ids: dict[str, int] = field(default_factory=dict)
    multi_masks: list[np.ndarray] = field(default_factory=list)
    output_video_path: str | None = None
    yolo_zip_path: str | None = None

    def video_state(self) -> dict:
        return {
            "user_name": self.id,
            "video_name": self.video_name,
            "original_video_path": self.original_video_path,
            "processed_video_path": self.processed_video_path,
            "transcode_info": self.transcode_info,
            "origin_images": self.origin_images,
            "painted_images": self.painted_images,
            "masks": self.masks,
            "logits": self.logits,
            "select_frame_number": self.select_frame_number,
            "fps": self.fps,
            "class_names": self.class_names,
            "mask_class_ids": self.mask_class_ids,
        }

    def interactive_state(self) -> dict:
        return {
            "track_end_number": self.track_end_number,
            "multi_mask": {"mask_names": self.mask_names, "masks": self.multi_masks},
            "class_names": self.class_names,
            "mask_class_ids": self.mask_class_ids,
        }


class ModelManager:
    def __init__(self) -> None:
        self._model: TrackingAnything | None = None
        self._lock = threading.Lock()

    def get(self) -> TrackingAnything:
        with self._lock:
            if self._model is None:
                self._model = self._load()
            return self._model

    def _load(self) -> TrackingAnything:
        device = os.environ.get("AUTOLABELER_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
        sam_model_type = os.environ.get("AUTOLABELER_SAM_MODEL_TYPE", "vit_h")
        checkpoint_dir = PROJECT_ROOT / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        sam_filename, sam_url = SAM_CHECKPOINTS[sam_model_type]
        xmem_filename, xmem_url = XMEM_CHECKPOINT
        sam_path = download_checkpoint(sam_url, checkpoint_dir / sam_filename)
        xmem_path = download_checkpoint(xmem_url, checkpoint_dir / xmem_filename)

        args = argparse.Namespace(device=device, sam_model_type=sam_model_type, mask_save=False, debug=False, port=8000)
        logger.info("Loading models on %s", device)
        return TrackingAnything(str(sam_path), str(xmem_path), None, args)


def download_checkpoint(url: str, path: Path) -> Path:
    if path.exists():
        return path
    logger.info("Downloading checkpoint %s", path.name)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    handle.write(chunk)
    return path


def as_numpy_image(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    return image


def image_response(image: Image.Image | np.ndarray) -> StreamingResponse:
    image = as_numpy_image(image)
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=92)
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/jpeg")


def session_summary(state: SessionState) -> dict:
    height, width = state.origin_images[0].shape[:2]
    mask_entries = [
        {
            "name": name,
            "class_id": state.mask_class_ids.get(name, 0),
            "class_name": state.class_names[state.mask_class_ids.get(name, 0)]
            if 0 <= state.mask_class_ids.get(name, 0) < len(state.class_names)
            else "",
        }
        for name in state.mask_names
    ]
    return {
        "session_id": state.id,
        "video_name": state.video_name,
        "frame_count": len(state.origin_images),
        "fps": state.fps,
        "width": width,
        "height": height,
        "selected_frame": state.select_frame_number,
        "mask_names": state.mask_names,
        "classes": state.class_names,
        "masks": mask_entries,
        "frame_url": f"/api/sessions/{state.id}/frames/{state.select_frame_number}?kind=painted&v={int(time.time() * 1000)}",
        "output_video_url": f"/api/sessions/{state.id}/output-video" if state.output_video_path else None,
        "yolo_zip_url": f"/api/sessions/{state.id}/yolo.zip" if state.yolo_zip_path else None,
    }


def get_session(session_id: str) -> SessionState:
    state = sessions.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return state


def normalize_class_name(class_name: str) -> str:
    cleaned = class_name.strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="Class name is required")
    return cleaned


def class_id_for_name(state: SessionState, class_name: str) -> int:
    class_name = normalize_class_name(class_name)
    if class_name not in state.class_names:
        state.class_names.append(class_name)
    return state.class_names.index(class_name)


def compose_template_mask(state: SessionState, mask_names: list[str]) -> np.ndarray:
    if state.multi_masks:
        selected = sorted(mask_names or [state.mask_names[0]])
        template_mask = np.zeros_like(state.multi_masks[0], dtype=np.uint8)
        for name in selected:
            try:
                mask_number = int(name.split("_")[1]) - 1
                template_mask = np.clip(template_mask + state.multi_masks[mask_number] * (mask_number + 1), 0, mask_number + 1)
            except (IndexError, ValueError):
                raise HTTPException(status_code=400, detail=f"Invalid mask name: {name}") from None
        state.masks[state.select_frame_number] = template_mask
        return template_mask
    return state.masks[state.select_frame_number]


def generate_video_from_frames(frames: list[Image.Image | np.ndarray], output_path: Path, fps: float) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    valid_frames = [as_numpy_image(frame) for frame in frames if frame is not None]
    if not valid_frames:
        raise HTTPException(status_code=400, detail="No frames to write")

    height, width = valid_frames[0].shape[:2]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), int(float(fps) or 30), (width, height))
    for frame in valid_frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    return str(output_path)


def paint_frame_mask(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask is None or len(np.unique(mask)) <= 1:
        return frame
    painted = frame.copy()
    for obj in range(1, int(mask.max()) + 1):
        if np.any(mask == obj):
            painted = mask_painter(painted, (mask == obj).astype("uint8"), mask_color=obj + 1)
    return as_numpy_image(painted)


app = FastAPI(title="AutoLabeler API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_manager = ModelManager()
model_lock = threading.Lock()
sessions: dict[str, SessionState] = {}


@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "cuda": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


@app.post("/api/sessions")
async def create_session(video: UploadFile = File(...)) -> dict:
    session_id = uuid.uuid4().hex
    upload_dir = PROJECT_ROOT / "temp_uploads"
    upload_dir.mkdir(exist_ok=True)
    filename = Path(video.filename or "upload.mp4").name
    upload_path = upload_dir / f"{session_id}_{filename}"

    with upload_path.open("wb") as handle:
        shutil.copyfileobj(video.file, handle)

    video_processor = get_video_processor()
    original_info = video_processor.get_video_info(str(upload_path))
    scale_ratio, target_width, target_height = video_processor.calculate_resize_ratio(
        original_info["original_width"],
        original_info["original_height"],
    )

    if scale_ratio < 1.0:
        processed_path, transcode_info = video_processor.transcode_video(str(upload_path))
    else:
        processed_path = str(upload_path)
        transcode_info = None

    frames: list[np.ndarray] = []
    cap = cv2.VideoCapture(processed_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or original_info["fps"] or 30.0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        raise HTTPException(status_code=400, detail="Could not extract frames from video")

    state = SessionState(
        id=session_id,
        video_name=filename,
        original_video_path=str(upload_path),
        processed_video_path=processed_path,
        transcode_info=transcode_info,
        origin_images=frames,
        painted_images=frames.copy(),
        masks=[np.zeros(frames[0].shape[:2], np.uint8) for _ in frames],
        logits=[None for _ in frames],
        fps=float(fps),
    )

    with model_lock:
        model = model_manager.get()
        model.samcontroler.sam_controler.reset_image()
        model.samcontroler.sam_controler.set_image(state.origin_images[0])

    sessions[session_id] = state
    summary = session_summary(state)
    summary["video_info"] = {
        "original_width": original_info["original_width"],
        "original_height": original_info["original_height"],
        "processed_width": target_width if transcode_info else original_info["original_width"],
        "processed_height": target_height if transcode_info else original_info["original_height"],
        "scale_ratio": scale_ratio,
    }
    return summary


@app.get("/api/sessions/{session_id}")
def read_session(session_id: str) -> dict:
    return session_summary(get_session(session_id))


@app.get("/api/sessions/{session_id}/frames/{frame_index}")
def read_frame(session_id: str, frame_index: int, kind: Literal["origin", "painted"] = "painted") -> StreamingResponse:
    state = get_session(session_id)
    if frame_index < 0 or frame_index >= len(state.origin_images):
        raise HTTPException(status_code=404, detail="Frame not found")
    if kind == "painted":
        frame = state.painted_images[frame_index]
        if frame is state.origin_images[frame_index] and len(np.unique(state.masks[frame_index])) > 1:
            frame = paint_frame_mask(state.origin_images[frame_index], state.masks[frame_index])
    else:
        frame = state.origin_images[frame_index]
    return image_response(frame)


@app.post("/api/sessions/{session_id}/select-frame")
def select_frame(session_id: str, payload: SelectFrameRequest) -> dict:
    state = get_session(session_id)
    if payload.frame_index < 0 or payload.frame_index >= len(state.origin_images):
        raise HTTPException(status_code=400, detail="Frame index out of range")
    state.select_frame_number = payload.frame_index
    state.click_points.clear()
    state.click_labels.clear()
    with model_lock:
        model = model_manager.get()
        model.samcontroler.sam_controler.reset_image()
        model.samcontroler.sam_controler.set_image(state.origin_images[payload.frame_index])
    return session_summary(state)


@app.post("/api/sessions/{session_id}/click")
def click_segment(session_id: str, payload: ClickRequest) -> dict:
    state = get_session(session_id)
    label = 1 if payload.label == "positive" else 0
    state.click_points.append([payload.x, payload.y])
    state.click_labels.append(label)

    with model_lock:
        model = model_manager.get()
        frame = state.origin_images[state.select_frame_number]
        model.samcontroler.sam_controler.reset_image()
        model.samcontroler.sam_controler.set_image(frame)
        mask, logit, painted_image = model.first_frame_click(
            image=frame,
            points=np.array(state.click_points),
            labels=np.array(state.click_labels),
            multimask=True,
        )

    state.masks[state.select_frame_number] = mask
    state.logits[state.select_frame_number] = logit
    state.painted_images[state.select_frame_number] = painted_image
    return session_summary(state)


@app.post("/api/sessions/{session_id}/clear-clicks")
def clear_clicks(session_id: str) -> dict:
    state = get_session(session_id)
    state.click_points.clear()
    state.click_labels.clear()
    state.painted_images[state.select_frame_number] = state.origin_images[state.select_frame_number]
    state.masks[state.select_frame_number] = np.zeros(state.origin_images[state.select_frame_number].shape[:2], np.uint8)
    state.logits[state.select_frame_number] = None
    return session_summary(state)


@app.post("/api/sessions/{session_id}/masks")
def add_mask(session_id: str, payload: AddMaskRequest) -> dict:
    state = get_session(session_id)
    mask = state.masks[state.select_frame_number]
    if len(np.unique(mask)) <= 1:
        raise HTTPException(status_code=400, detail="No mask on selected frame")
    state.multi_masks.append(mask.copy())
    name = f"mask_{len(state.multi_masks):03d}"
    state.mask_names.append(name)
    state.mask_class_ids[name] = class_id_for_name(state, payload.class_name)
    state.click_points.clear()
    state.click_labels.clear()

    frame = state.origin_images[state.select_frame_number].copy()
    for index, stored_mask in enumerate(state.multi_masks):
        frame = mask_painter(frame, stored_mask.astype("uint8"), mask_color=index + 2)
    state.painted_images[state.select_frame_number] = frame
    return session_summary(state)


@app.patch("/api/sessions/{session_id}/masks/{mask_name}")
def update_mask_class(session_id: str, mask_name: str, payload: UpdateMaskRequest) -> dict:
    state = get_session(session_id)
    if mask_name not in state.mask_names:
        raise HTTPException(status_code=404, detail="Mask not found")
    state.mask_class_ids[mask_name] = class_id_for_name(state, payload.class_name)
    return session_summary(state)


@app.delete("/api/sessions/{session_id}/masks")
def clear_masks(session_id: str) -> dict:
    state = get_session(session_id)
    state.mask_names.clear()
    state.mask_class_ids.clear()
    state.multi_masks.clear()
    state.click_points.clear()
    state.click_labels.clear()
    state.painted_images[state.select_frame_number] = state.origin_images[state.select_frame_number]
    return session_summary(state)


@app.post("/api/sessions/{session_id}/track")
def track_video(session_id: str, payload: TrackRequest) -> dict:
    state = get_session(session_id)
    if payload.xmem_size is not None and payload.xmem_size not in {0, 360, 480, 720}:
        raise HTTPException(status_code=400, detail="Invalid tracking precision")
    state.track_end_number = payload.end_frame
    end_frame = payload.end_frame if payload.end_frame is not None else len(state.origin_images)
    end_frame = max(state.select_frame_number + 1, min(end_frame, len(state.origin_images)))
    template_mask = compose_template_mask(state, payload.mask_names)
    if len(np.unique(template_mask)) <= 1:
        raise HTTPException(status_code=400, detail="Please add at least one mask before tracking")

    output_path = PROJECT_ROOT / "result" / "track" / f"{session_id}_{Path(state.video_name).stem}.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    first_frame = as_numpy_image(state.origin_images[0])
    height, width = first_frame.shape[:2]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), int(float(state.fps) or 30), (width, height))

    try:
        with model_lock:
            model = model_manager.get()
            logger.info(
                "Start tracking session=%s frames=%s-%s xmem_size=%s",
                session_id,
                state.select_frame_number,
                end_frame - 1,
                payload.xmem_size or "original",
            )
            model.xmem.set_tracking_size(payload.xmem_size or 0)
            model.xmem.clear_memory()
            try:
                for frame_index, frame in tqdm(enumerate(state.origin_images), total=len(state.origin_images), desc="Tracking image"):
                    if frame_index < state.select_frame_number or frame_index >= end_frame:
                        output_frame = as_numpy_image(state.painted_images[frame_index])
                    else:
                        if frame_index == state.select_frame_number:
                            mask, _logit, painted_image = model.xmem.track(frame, template_mask)
                        else:
                            mask, _logit, painted_image = model.xmem.track(frame)
                        state.masks[frame_index] = mask
                        state.logits[frame_index] = None
                        state.painted_images[frame_index] = state.origin_images[frame_index]
                        output_frame = as_numpy_image(painted_image)
                    writer.write(cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR))
            finally:
                writer.release()
                model.xmem.clear_memory()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise HTTPException(
                status_code=507,
                detail="CUDA out of memory while tracking. Restart the backend and lower AUTOLABELER_XMEM_SIZE, or track a shorter frame range.",
            ) from exc
        raise

    state.output_video_path = str(output_path)
    return session_summary(state)


@app.post("/api/sessions/{session_id}/yolo")
def export_yolo(session_id: str, payload: YoloRequest) -> dict:
    state = get_session(session_id)
    state.track_end_number = payload.end_frame
    output_dir, _operation_log = get_sampled_generator().generate_yolo_dataset_sampled(
        state.video_state(),
        state.interactive_state(),
        payload.mask_names,
    )
    if not output_dir:
        raise HTTPException(status_code=400, detail="Failed to generate YOLO dataset")

    zip_path = Path(f"{output_dir}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in Path(output_dir).rglob("*"):
            if file_path.is_file():
                zipf.write(file_path, file_path.relative_to(output_dir))
    state.yolo_zip_path = str(zip_path)
    return session_summary(state)


@app.get("/api/sessions/{session_id}/output-video")
def output_video(session_id: str) -> FileResponse:
    state = get_session(session_id)
    if not state.output_video_path or not Path(state.output_video_path).exists():
        raise HTTPException(status_code=404, detail="Output video not found")
    return FileResponse(state.output_video_path, media_type="video/mp4", filename=Path(state.output_video_path).name)


@app.get("/api/sessions/{session_id}/yolo.zip")
def yolo_zip(session_id: str) -> FileResponse:
    state = get_session(session_id)
    if not state.yolo_zip_path or not Path(state.yolo_zip_path).exists():
        raise HTTPException(status_code=404, detail="YOLO archive not found")
    return FileResponse(state.yolo_zip_path, media_type="application/zip", filename=Path(state.yolo_zip_path).name)
