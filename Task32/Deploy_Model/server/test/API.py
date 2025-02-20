import requests
import numpy as np
import json

# Địa chỉ Triton Server (chạy local)
TRITON_SERVER_URL = "http://localhost:8000/v2/models/densenet_onnx"

# Chuẩn bị dữ liệu input (Ảnh random [1, 3, 224, 224])
input_data = np.random.rand(1, 3, 224, 224).astype(np.float32)

# Chuẩn bị payload gửi API
payload = {
    "inputs": [
        {
            "name": "data_0",
            "shape": list(input_data.shape),
            "datatype": "FP32",
            "data": input_data.tolist()
        }
    ]
}

# Gửi request POST đến Triton
response = requests.post(TRITON_SERVER_URL, json=payload)

# Kiểm tra kết quả
if response.status_code == 200:
    result = response.json()
    print("✅ Inference Done! Kết quả:", result)
else:
    print("❌ Lỗi inference:", response.text)
