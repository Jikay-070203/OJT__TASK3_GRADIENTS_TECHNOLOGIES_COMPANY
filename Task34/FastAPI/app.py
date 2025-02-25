from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
from trism import TritonModel
from PIL import Image
import io

# Khởi tạo FastAPI
app = FastAPI()

# Cấu hình Triton Server
TRITON_SERVER_URL = "localhost:8000"
MODEL_NAME = "densenet_onnx"

# Load model từ Triton Server (dùng trism)
model = TritonModel(
    model=MODEL_NAME,
    url=TRITON_SERVER_URL,
    version=1,
    grpc=False  # use HTTP
)

@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    try:
        # Đọc file ảnh
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Tiền xử lý ảnh
        img = img.resize((224, 224))
        img_array = np.array(img).astype(np.float32)
        img_array = np.transpose(img_array, (2, 0, 1))  # Đổi thành (C, H, W)
        img_array /= 255.0  # Chuẩn hóa về [0,1]

        # Gửi request đến Triton
        outputs = model.run(data=[img_array])  # Truyền dưới dạng list

        # Debug output
        print("Output Keys:", outputs.keys())  # Debug key output
        print("Model Output:", outputs["fc6_1"])  # Lấy giá trị đầu ra

        # Kiểm tra output của Triton
        if "fc6_1" not in outputs:
            raise ValueError(f"Output 'fc6_1' not found in model output: {outputs.keys()}")

        # Lấy kết quả từ mô hình
        inference_output = outputs["fc6_1"]
        predicted_class = np.argmax(inference_output)
        confidence = float(inference_output[predicted_class])

        return JSONResponse(content={
            "predicted_class": int(predicted_class),
            "confidence": confidence
        })

    except Exception as e:
        print(f"🔥 Error: {str(e)}")  
        return JSONResponse(content={"error": str(e)}, status_code=500)
