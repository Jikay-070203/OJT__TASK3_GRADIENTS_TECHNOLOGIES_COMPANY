import requests
import numpy as np
import json

# Tạo dữ liệu đầu vào giả lập (3x224x224)
input_data = np.random.rand(3, 224, 224).astype(np.float32)  # Đúng: shape [3,224,224]


# Định dạng payload JSON theo Triton
payload = {
    "inputs": [
        {
            "name": "data_0",
            "shape": input_data.shape,  # Đảm bảo đúng shape [3,224,224]
            "datatype": "FP32",
            "data": input_data.tolist()
        }
    ]
}

# Gửi request đến Triton Server
url = "http://localhost:8000/v2/models/densenet_onnx/infer"
response = requests.post(url, json=payload)

# In kết quả
print(response.json())
