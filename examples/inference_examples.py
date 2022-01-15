from trtoptimize.inference import Inference
import numpy as np

infer=Inference()
model_name="mobilenet_model"
url="127.0.0.1:8000"
infer.add_inference_model(model_name, url)                                    # Add model name and url 

batch_size = 8
batched_input = np.ones((batch_size, 224, 224, 3), dtype=np.float32)          # create batched input
infer.inference(batched_input)
