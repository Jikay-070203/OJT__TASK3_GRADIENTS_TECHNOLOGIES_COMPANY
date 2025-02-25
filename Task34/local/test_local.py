from trism import TritonModel
import numpy as np

# Create triton model.
model = TritonModel(
  model="densenet_onnx", 
  version=0,            
  url="localhost:8001", 
  grpc=True             # Use gRPC or Http.
)

# View metadata.
for inp in model.inputs:
  print(f"name: {inp.name}, shape: {inp.shape}, datatype: {inp.dtype}\n")
for out in model.outputs:
  print(f"name: {out.name}, shape: {out.shape}, datatype: {out.dtype}\n")

# data input ảnh 
img_array = np.random.rand(3, 224, 224).astype(np.float32)  #shape [3,224,224]

# Gửi dữ liệu đến Triton
outputs = model.run(data=[img_array])  # Truyền dạng list

# Kiểm tra output
print("Output Keys:", outputs.keys())  # Debug key output
print("Model Output:", outputs["fc6_1"])  # Lấy giá trị đầu ra

# Lấy class có xác suất cao nhất
predicted_class = np.argmax(outputs["fc6_1"])
print("Predicted Class:", predicted_class)
