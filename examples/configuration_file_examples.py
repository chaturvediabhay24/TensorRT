from trtoptimize.optimize import TRTOptimize

optimizer=TRTOptimize()
# With minimum params
optimizer.create_config_file_tf("mobilenet_saved_model")
# With all params
optimizer.create_config_file_tf(saved_model_dir="mobilenet_saved_model",
                                enable_dynamic_batching=True,
                                prefered_batch_size=[2,4,6,8], 
                                no_of_instance=2, 
                                enable_gpu_optimization=True)
