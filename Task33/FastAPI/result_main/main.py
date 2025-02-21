from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import Response
from pydantic import BaseModel
import grpc
import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.grpc import InferInput, InferRequestedOutput
import cv2
import io
import PIL.Image as Image

# Địa chỉ Triton Server
TRITON_GRPC_URL = "172.17.0.2:8001"  # Cổng gRPC

# Tạo app FastAPI
app = FastAPI()

# Tạo client Triton (khởi tạo một lần và tái sử dụng)
triton_client = grpcclient.InferenceServerClient(url=TRITON_GRPC_URL)

# Schema cho input số liệu
class InputData(BaseModel):
    data: list  # input phải là list số thực

# API xử lý ảnh
@app.post("/infer/")
async def infer_image(file: UploadFile = File(...)):
    """
    Nhận file ảnh, xử lý inference và trả về ảnh đã xử lý.
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = np.array(image)
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Chuyển thành ảnh grayscale
        _, encoded_image = cv2.imencode(".png", gray_image)
        return Response(content=encoded_image.tobytes(), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")