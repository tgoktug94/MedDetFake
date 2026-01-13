# main_api.py
# FastAPI - Only:
#  - 8-class synthesis detection
#  - density-map based manipulation detection (boxes + density map + annotated image)

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import os
import tempfile
import uuid
import base64
import numpy as np
import cv2
from PIL import Image

from main_interface import (
    detect_fake_synthesis,          # existing 8-class
    detect_manipulated_density      # new density-based
)

# --------------------------------------------------------------------
# APP
# --------------------------------------------------------------------
app = FastAPI(title="MedXFake Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/templates", StaticFiles(directory="templates"), name="templates")

IMG_SIZE = 128  # model expects 128x128

# --------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------
def save_upload_to_temp(upload: UploadFile) -> str:
    suffix = os.path.splitext(upload.filename or "")[1] or ".png"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(tmp_path, "wb") as f:
        f.write(upload.file.read())
    return tmp_path

def read_gray_resize_128(path: str):
    img = Image.open(path).convert("L").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype(np.uint8)
    return arr  # (128,128) uint8

def save_annotated_boxes(gray_u8: np.ndarray, boxes_xyxy, out_path: str):
    # draw on BGR for colored boxes
    bgr = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
    for (x1,y1,x2,y2) in boxes_xyxy:
        cv2.rectangle(bgr, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.imwrite(out_path, bgr)

def save_density_colormap(density_u8: np.ndarray, out_path: str):
    # density_u8: 0-255
    heat = cv2.applyColorMap(density_u8, cv2.COLORMAP_JET)
    cv2.imwrite(out_path, heat)

def english_report(boxes_xyxy):
    if len(boxes_xyxy) == 0:
        return "No manipulated region was detected."
    lines = ["Manipulated regions detected at:"]
    for i,(x1,y1,x2,y2) in enumerate(boxes_xyxy, start=1):
        lines.append(f"  Region {i}: (x1={x1}, y1={y1}, x2={x2}, y2={y2})")
    return "\n".join(lines)

# --------------------------------------------------------------------
# ROOT PAGE (optional)
# --------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def root_page():
    index_path = "templates/index.html"
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>MedXFake Detection API</h1>", status_code=200)

# --------------------------------------------------------------------
# 1) MANIPULATION DETECTION (Density-map + Boxes + Annotated + Heatmap)
# --------------------------------------------------------------------
@app.post("/detect/manipulation_density")
async def api_detect_manipulation_density(
    model_path: str = Form(...),
    thr: float = Form(0.28),
    min_area: int = Form(20),
    image: UploadFile = File(...)
):
    """
    Returns:
      - annotated image with predicted bounding boxes
      - density heatmap image
      - bbox coordinates
      - English message
    """
    try:
        img_path = save_upload_to_temp(image)

        # inference (density + boxes)
        result = detect_manipulated_density(
            model_path=model_path,
            image_path=img_path,
            thr=thr,
            min_area=min_area
        )

        boxes = result["boxes_xyxy"]
        density_u8 = result.get("density_u8", None)

        # base image (128x128 gray)
        gray_u8 = read_gray_resize_128(img_path)

        # save annotated + density heatmap to /static
        ann_name = f"manip_boxes_{uuid.uuid4().hex}.png"
        den_name = f"density_{uuid.uuid4().hex}.png"

        ann_path = os.path.join("static", ann_name)
        den_path = os.path.join("static", den_name)

        save_annotated_boxes(gray_u8, boxes, ann_path)

        if density_u8 is None:
            density_u8 = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        save_density_colormap(density_u8, den_path)

        message = english_report(boxes)

        return JSONResponse({
            "status": "ok",
            "is_manipulated": result["is_manipulated"],
            "count": result["count"],
            "boxes_xyxy": boxes,
            "threshold": result["thr"],
            "min_area": result["min_area"],
            "device": result["device"],
            "message_en": message,
            "annotated_boxes_image_url": f"/static/{ann_name}",
            "density_map_image_url": f"/static/{den_name}",
        })

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

# --------------------------------------------------------------------
# 2) 8-Class SYNTHESIS DETECTION (keep as-is)
# --------------------------------------------------------------------
@app.post("/detect/synthesis8")
async def api_detect_synthesis8(
    model_path: str = Form(...),
    image: UploadFile = File(...)
):
    """
    8 classes: [brain_real, brain_fake, chest_real, chest_fake, kidney_real, ...]
    """
    try:
        img_path = save_upload_to_temp(image)
        result = detect_fake_synthesis(
            model_path=model_path,
            image_path=img_path
        )
        return JSONResponse({"status": "ok", "result": result})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
