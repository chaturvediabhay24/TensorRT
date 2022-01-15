from trtoptimize.optimize import TRTOptimize
import numpy as np

# Using FP32 as default precision
optimizer=TRTOptimize()
saved_model_dir="mobilenet_saved_model"
optimizer.optimize(saved_model_dir, version='tf1')

# Using FP16 precision
optimizer=TRTOptimize()
saved_model_dir = "mobilenet_saved_model"
optimizer.optimize( saved_model_dir, version='tf1', precision="FP16")

# Using INT8 precision (Needs callibration dataset)
optimizer=TRTOptimize()

# Creating callibration dataset
batch_size = 8
batched_input = np.ones((batch_size, 224, 224, 3), dtype=np.float32)
saved_model_dir = "mobilenet_saved_model"
optimizer.optimize( saved_model_dir, version='tf1', precision="INT8", calibration_data= batched_input)
