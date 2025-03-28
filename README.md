# 🚀 Model Deployment Pipeline with ONNX, FastAPI & Trism

This project demonstrates a complete pipeline for deploying deep learning models using **ONNX**, **Triton Inference Server**, **FastAPI**, and the lightweight **Trism** alternative and Automatically download model from **Hugging Face** on container startup.

---

## 📁 Project Structure

```
Deploy_model/
│
├── Task_1_ONNX_Model_Conversion_&_Super_Resolution/
│   └── Convert models to ONNX, serve with Triton
│
├── Task_2_Deploy_Model/
│   └── Run Triton Inference Server, benchmark performance
│
├── Task_3_FastAPI/
│   └── Serve model with FastAPI, call Triton via gRPC
│
├── Task_4_FastAPI_with_Trism/
│   └── Replace TritonClient with Trism, Automatically download model from **Hugging Face** on container startup.
```

---

## ✅ Task 1: ONNX Model Conversion & Super Resolution

- Convert PyTorch models (e.g., Super Resolution) to ONNX format.
- Prepare model repository for Triton serving.
- Validate the ONNX outputs to ensure correctness.

---

## ✅ Task 2: Deploy Model with Triton

- Pull Triton Inference Server image from NGC.
- Launch Triton with the model repository.
- Send inference requests and test API functionality.
- Use **Triton Performance Analyzer** to benchmark:
  - Throughput
  - Latency
  - Concurrency

---

## ✅ Task 3: FastAPI with Triton (gRPC) and HTTP

- Build a **FastAPI** application to:
  - Accept input via REST API.
  - Send inference requests to Triton via gRPC (`tritonclient.grpc`) or HTTP (`tritonclient.grpc`)
  - Return predictions to users.

```bash
# Start server
uvicorn main:app --reload
```

- Sample route:

```python
@app.post("/predict_grpc/")
def predict_grpc(data: dict): ...
```

---

## ✅ Task 4: FastAPI with Trism + Hugging Face

- Replace `tritonclient` with **Trism**: a lightweight wrapper for Triton inference.
- Use Docker image `hieupth/tritonserver` (lightweight).
- Automatically download model from **Hugging Face** on container startup.
- Improve deployment efficiency and reduce image size.

> ⚠️ Notes:
>
> - Handle port conflicts during server startup.
> - Adjust model structure on Hugging Face to match Triton input expectations.

---

## 🛠️ Tech Stack

- Python 3.8+
- PyTorch, ONNX
- Triton Inference Server
- FastAPI, Uvicorn
- gRPC or http, Trism
- Docker & Docker Compose
- Hugging Face Model Hub

---

## 📌 Future Work

- Optimize port management & Docker networking.
- Align Hugging Face model format with Triton specs.
- Test inference performance and push final setup to GitHub.
