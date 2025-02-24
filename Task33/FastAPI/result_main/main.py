from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.grpc import InferInput, InferRequestedOutput

import cv2
import io
import PIL.Image as Image

app = FastAPI()

# Địa chỉ Triton Server (gRPC endpoint)
TRITON_GRPC_URL = "172.17.0.2:8001"  
MODEL_NAME = "densenet_onnx"

#input/output
INPUT_NAME = "data_0"
OUTPUT_NAME = "fc6_1"
INPUT_WIDTH = 224
INPUT_HEIGHT = 224

# Tạo client Triton 
triton_client = grpcclient.InferenceServerClient(url=TRITON_GRPC_URL)

@app.post("/infer")
async def infer_image(file: UploadFile = File(...)):
    try:
        # Đọc dữ liệu ảnh từ UploadFile
        image_bytes = await file.read()

        # Đọc ảnh bằng PIL, chuyển RGB
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Resize ảnh về (224,224)
        pil_img = pil_img.resize((INPUT_WIDTH, INPUT_HEIGHT))

        # Chuyển sang NumPy array, shape (224,224,3)
        np_img = np.array(pil_img, dtype=np.float32)

        # Chuyển (H,W,C) => (C,H,W) => (3,224,224)
        np_img = np.transpose(np_img, (2, 0, 1))

        # Thêm batch dimension => (1,3,224,224)
        np_img = np.expand_dims(np_img, axis=0)

        # Chuẩn hóa ảnh (nếu cần), ví dụ chia 255
        np_img /= 255.0

        # Tạo InferInput cho Triton
        infer_input = InferInput(INPUT_NAME, np_img.shape, "FP32")
        infer_input.set_data_from_numpy(np_img)

        # Yêu cầu output
        infer_output = InferRequestedOutput(OUTPUT_NAME)

        # Gửi request đến Triton
        result = triton_client.infer(
            model_name=MODEL_NAME,
            inputs=[infer_input],
            outputs=[infer_output]
        )

        # Lấy output data (theo config, shape [1,1000,1,1])
        output_data = result.as_numpy(OUTPUT_NAME)
        # Reshape về [1000]
        output_data = output_data.reshape(-1)

        # Tìm lớp dự đoán (argmax)
        predicted_class = int(np.argmax(output_data))
        confidence = float(output_data[predicted_class])

        # Trả về JSON: class, confidence
        return JSONResponse(content={
            "predicted_class": predicted_class,
            "confidence": confidence
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
