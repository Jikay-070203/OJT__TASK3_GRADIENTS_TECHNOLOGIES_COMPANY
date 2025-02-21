import numpy as np
import requests
import json

# URL của Triton Inference Server
TRITON_SERVER_URL = "http://localhost:8000/v2/models/densenet_onnx/infer"

# Tạo dữ liệu đầu vào (chú ý định dạng đúng)
input_data = np.random.rand(3, 224, 224).astype(np.float32)  # Không có batch dimension

payload = {
    "inputs": [
        {
            "name": "data_0",
            "shape": [3, 224, 224],  # Đúng định dạng model yêu cầu
            "datatype": "FP32",
            "data": input_data.tolist(),
        }
    ]
}

# Gửi request đến Triton Inference Server
response = requests.post(TRITON_SERVER_URL, json=payload)

# Kiểm tra response
if response.status_code == 200:
    response_json = response.json()  # Chuyển đổi sang JSON
    output_data = response_json["outputs"][0]["data"]  # Lấy dữ liệu đầu ra
    predicted_class = np.argmax(output_data)  # Tìm class có xác suất cao nhất

    print(f"Predicted class: {predicted_class}")
else:
    print(f"Request failed with status code {response.status_code}")
    print(response.text)  # In ra lỗi nếu có
