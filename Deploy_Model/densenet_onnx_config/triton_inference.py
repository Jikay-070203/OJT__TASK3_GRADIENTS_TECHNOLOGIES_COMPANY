import tritonclient.http as httpclient
import numpy as np

# Triton server details 
TRITON_SERVER_URL = "localhost:8000"
MODEL_NAME = "densenet_onnx"
INPUT_NAME = "data_0"
OUTPUT_NAME = "fc6_1"

# create client Triton HTTP (keep)
triton_client = httpclient.InferenceServerClient(url=TRITON_SERVER_URL, verbose=False)

# 1. create data input (fixed size)
input_data = np.random.randn(3, 224, 224).astype(np.float32)  # delete size batch

#2. create obj InferInput
inputs = []
outputs = []
input_tensor = httpclient.InferInput(INPUT_NAME, input_data.shape, "FP32")

# 3. setup data for input tensor 
input_tensor.set_data_from_numpy(input_data)

inputs.append(input_tensor)
outputs.append(httpclient.InferRequestedOutput(OUTPUT_NAME))

# create Inference
try:
    results = triton_client.infer(
        model_name=MODEL_NAME,
        inputs=inputs,
        outputs=outputs
    )

    # result
    output_data = results.as_numpy(OUTPUT_NAME)
    print(f"Kết quả suy luận: {output_data}")

except httpclient.InferenceServerException as e:
    print("Suy luận thất bại: " + str(e))