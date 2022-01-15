### How to use the library?

clone the directory:

```shell
git clone <Repository Name>
```
Install the dependencies.
```shell
pip install setup.py
```
### Perform Optimization on TF2.x Models

Using FP32 precision
```shell
lib=Library()
lib.optimize("mobilenet_saved_model")
```

Using FP16 precision
```shell
lib=Library()
lib.optimize("mobilenet_saved_model", precision="FP16")
```

Using INT8 precision
```shell
# Creating callibration dataset
batch_size = 8
batched_input = np.ones((batch_size, 224, 224, 3), dtype=np.float32)
lib=Library()
lib.optimize("mobilenet_saved_model", precision="int8", calibration_data= batched_input)
```
### Perform Optimization on TF1.x Models

Using FP32 precision
```shell
lib=Library()
lib.optimize("mobilenet_saved_model", version='tf1')
```

Using FP16 precision
```shell
lib=Library()
lib.optimize("mobilenet_saved_model", precision="FP16", version='tf1')
```

Using INT8 precision
```shell
# Creating callibration dataset
batch_size = 8
batched_input = np.ones((batch_size, 224, 224, 3), dtype=np.float32)
lib=Library()
lib.optimize("mobilenet_saved_model", precision="int8", calibration_data= batched_input, version='tf1')
```

### How to perform inference

```shell
from trtoptimize.inference import Inference
import numpy as np

infer=Inference()
model_name="mobilenet_model"
url="127.0.0.1:8000"
infer.add_inference_model(model_name, url)                                    # Add model name and url 

batch_size = 8
batched_input = np.ones((batch_size, 224, 224, 3), dtype=np.float32)          # create batched input
infer.inference(batched_input)
```

### How to create configuration file

With minimum params
```shell
from trtoptimize.optimize import TRTOptimize

optimizer=TRTOptimize()
optimizer.create_config_file_tf("mobilenet_saved_model")
```

With all params
```shell
optimizer.create_config_file_tf(saved_model_dir="mobilenet_saved_model",
                                enable_dynamic_batching=True,
                                prefered_batch_size=[2,4,6,8], 
                                no_of_instance=2, 
                                enable_gpu_optimization=True)
```
