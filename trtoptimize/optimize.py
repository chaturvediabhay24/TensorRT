from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.tools import saved_model_utils

class TRTOptimize:
    def __init__(self):
        self.enable_gpu_optimization='''
        optimization {{ execution_accelerators {{
            gpu_execution_accelerator : [ {{
            name : "tensorrt"
            parameters {{ key: "precision_mode" value: "{}" }}}}]
        }}}}
            '''

        self.instance_group='''
        instance_group [ {{ count: {} }}]
            '''

        self.prefered_batch_size='''
        dynamic_batching {{ prefered_batch_size:  {}  }}
            '''

        self.dynamic_batching='''
        dynamic_batching {}
            '''

        self.content='''\
        name: "{}"
        platform: "{}"
        max_batch_size: {}
        input [
            {{
                name: "{}"
                data_type: {}
                dims: {}
            }}
        ]
        output [
            {{
                name: "{}"
                data_type: {}
                dims: {}
            }}
        ]\
            '''
    def optimize_tf1(self, 
                saved_model_dir,
                precision='FP32', 
                max_workspace_size_in_bytes=8000000000, 
                calibration_data=None):
        
        saved_model_dir=self.get_saved_model_format(saved_model_dir)
        
        if(precision=='FP32'):
            converter = trt.TrtGraphConverter(
            input_saved_model_dir=saved_model_dir,
            precision_mode=trt.TrtPrecisionMode.FP32)
            converted_graph_def = converter.convert()
            output_saved_model_dir=saved_model_dir+'_TFTRT_{}'.format(precision)
            converter.save(output_saved_model_dir)
            print('Done Converting to TF-TRT {}'.format(precision))
        elif(precision=='FP16'):
            converter = trt.TrtGraphConverter(
            input_saved_model_dir=saved_model_dir,
            precision_mode=trt.TrtPrecisionMode.FP16)
            converted_graph_def = converter.convert()
            output_saved_model_dir=saved_model_dir+'_TFTRT_{}'.format(precision)
            converter.save(output_saved_model_dir)
            print('Done Converting to TF-TRT {}'.format(precision))
        elif(precision=='INT8'):
            print('Converting to TF-TRT INT8...')
            conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
                precision_mode=trt.TrtPrecisionMode.INT8, 
                max_workspace_size_bytes=max_workspace_size_in_bytes, 
                use_calibration=True,
                is_dynamic_op=True)
            converter = trt.TrtGraphConverter(
                input_saved_model_dir=saved_model_dir, 
                conversion_params=conversion_params)

            def my_calibration_input_fn():
                yield (calibration_data, )
            converter.convert(calibration_input_fn=my_calibration_input_fn)
            converter.save(output_saved_model_dir=saved_model_dir+'_TFTRT_{}'.format(precision))
            print('Done Converting to TF-TRT INT8')
            return
        else:
            print("Invalid precision mode!")
    def optimize_pytorch(self, 
                saved_model_dir, 
                precision='FP32', 
                max_workspace_size_in_bytes=8000000000, 
                calibration_data=None):
        pass
    
    def optimize(self, 
                saved_model_dir, 
                version="tf2",
                precision='FP32', 
                max_workspace_size_in_bytes=8000000000, 
                calibration_data=None):
        """
        Function helps in performing TensorRT optimizations on tensorflow models.


        Parameters
        ----------
        saved_model_dir : string
            Name of directory of your tensorflow saved model which needs to be optimized.
            
        version : {'TF1', 'TF2', 'PYTORCH'}, default 
            The version of model, which you want to optimize.

        precision : {'INT8', 'FP16', 'FP32'}, default: 'FP32'
            precision value to be used for optimizing models.

        max_workspace_size_in_bytes : int, default: 8000000000(8GB)
            maximum amount of workspace size in bytes.

        calibration_data : list/numpy_array, default: None
            Batched input data(representative training data) to perform precision calibration.
            This will only be used while performing INT8 precision.

        """
        if(version=='tf1'):
            self.optimize_tf1(self, 
                saved_model_dir, 
                precision='FP32', 
                max_workspace_size_in_bytes=8000000000, 
                calibration_data=None)
            return
        elif(version=='pytorch'):
            self.optimize_pytorch(self, 
                saved_model_dir, 
                precision='FP32', 
                max_workspace_size_in_bytes=8000000000, 
                calibration_data=None)
            return
        
        saved_model_dir=self.get_saved_model_format(saved_model_dir)
        
        if(precision=='FP32'):
            print('Converting to TF-TRT {}...'.format(precision))
            conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
            precision_mode=trt.TrtPrecisionMode.FP32,
            max_workspace_size_bytes=max_workspace_size_in_bytes)
            converter = trt.TrtGraphConverterV2(
            input_saved_model_dir=saved_model_dir, conversion_params=conversion_params)
            converter.convert()
            converter.save(output_saved_model_dir=saved_model_dir+'_TFTRT_{}'.format(precision))
            print('Done Converting to TF-TRT {}'.format(precision))
        elif(precision=='FP16'):
            print('Converting to TF-TRT {}...'.format(precision))
            conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
            precision_mode=trt.TrtPrecisionMode.FP16,
            max_workspace_size_bytes=max_workspace_size_in_bytes)
            converter = trt.TrtGraphConverterV2(
            input_saved_model_dir=saved_model_dir, conversion_params=conversion_params)
            converter.convert()
            converter.save(output_saved_model_dir=saved_model_dir+'_TFTRT_{}'.format(precision))
            print('Done Converting to TF-TRT {}'.format(precision))
        elif(precision=='INT8'):
            print('Converting to TF-TRT INT8...')
            conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
                precision_mode=trt.TrtPrecisionMode.INT8, 
                max_workspace_size_bytes=max_workspace_size_in_bytes, 
                use_calibration=True,
                is_dynamic_op=True)
            converter = trt.TrtGraphConverterV2(
                input_saved_model_dir=saved_model_dir, 
                conversion_params=conversion_params)

            def my_calibration_input_fn():
                yield (calibration_data, )
            converter.convert(calibration_input_fn=my_calibration_input_fn)
            converter.save(output_saved_model_dir=saved_model_dir+'_TFTRT_{}'.format(precision))
            print('Done Converting to TF-TRT INT8')
            return

        else:
            print("{} precision value not supported, Input either 'FP32', 'FP16' or 'INT8' as precision values".format(precision))
            print ("Invalid 'precision' value.")
            return
    
    def get_input_output_info(self, saved_model_dir):
        """
        Function helps in getting input, output related details of models.


        Parameters
        ----------
        saved_model_dir : string
            Name of directory of tensorflow saved model.

        Returns
        -------
        input_dimension, output_dimension : list 
            list containing dimensions of model input and output.

        input_name, output_name : string
            A string representing names of model input and output.        

        """
        tag_sets = saved_model_utils.get_saved_model_tag_sets(saved_model_dir)
        meta_graph_def = saved_model_utils.get_meta_graph_def(saved_model_dir, tag_sets[0][0])
        meta_graph = saved_model_utils.get_meta_graph_def(saved_model_dir, tag_sets[0][0])
        signature_def_map = meta_graph.signature_def
        keys=[i for i in signature_def_map.keys() if i != "__saved_model_init_op"]
        # print((keys))
        signature_def_key=keys[-1]
        inputs_tensor_info = meta_graph_def.signature_def[signature_def_key].inputs
        outputs_tensor_info = meta_graph_def.signature_def[signature_def_key].outputs
        info=[]
        for input_name, input_tensor in sorted(inputs_tensor_info.items()):
          input_dims = [int(dim.size) for dim in input_tensor.tensor_shape.dim]
          info.append( input_name)
          info.append(input_dims[1:])
        for output_name, output_tensor in sorted(outputs_tensor_info.items()):
          output_dims = [int(dim.size) for dim in output_tensor.tensor_shape.dim]
          info.append( output_name)
          info.append(output_dims[1:])
        return info[0], info[1], info[2], info[3]
    
    def create_config_file_torch(self,
                            model_name,
                            input_dims,
                            output_dims,
                            input_name,
                            output_name,
                            save_platform="pytorch_libtorch",
                            data_type='TYPE_FP32',
                            enable_dynamic_batching=False, 
                            prefered_batch_size=[], 
                            no_of_instance=1,
                            max_batch_size=128):
        """
        Function helps in creation of configuration file required while 
        deploying model on tensorrt inference server.


        Parameters
        ----------
        model_name : string
            Name that needs to be given to pytorch model.

        input_dimension, output_dimension : list 
            list containing dimensions of model input and output.

        input_name, output_name : string
            A string representing names of model input and output.

        save_platform : string, default: "pytorch_libtorch"
            platform for which it should be saved.

        data_type : {"TYPE_FP32", "TYPE_BOOL", "TYPE_UINT8", 
                    "TYPE_UINT16", "TYPE_UINT32", "TYPE_UINT64",
                    "TYPE_INT8", "TYPE_INT16", "TYPE_INT32", 
                    "TYPE_INT64", "TYPE_FP16", "TYPE_FP32", 
                    "TYPE_FP64", "TYPE_STRING"}, default: "TYPE_FP32"

        enable_dynamic_batching : bool, default: False
            parameter to enable dynamic batching while inferencing the model.

        prefered_batch_size : list, default=[]
            If dynamic batching is enabled what should be the preference for batch size.

        no_of_instance : int, default: 1
            Represents the no of instance/copies of model you want to create for server.

        max_batch_size : int, default: 128
            maximum allowed batch size for inference.

        """

        with open("config.pbtxt", "w") as file:
            file.write((self.content.format(saved_model, 
                                            save_platform, 
                                            max_batch_size, 
                                            input_name, 
                                            data_type, 
                                            input_dim, 
                                            output_name, 
                                            data_type, 
                                            output_dim)))

        if(enable_dynamic_batching and prefered_batch_size):
            with open("config.pbtxt", "a") as file:
                file.write(self.prefered_batch_size.format(prefered_batch_size))

        elif(enable_dynamic_batching and (not prefered_batch_size)):
            with open("config.pbtxt", "a") as file:
                file.write(self.dynamic_batching)

        if(no_of_instance>1):
            with open("config.pbtxt", "a") as file:
                file.write(self.instance_group.format(no_of_instance))
    
    def create_config_file_tf(self, 
                            saved_model_dir, 
                            data_type='TYPE_FP32',
                            save_platform="tensorflow_savedmodel",
                            enable_dynamic_batching=False, 
                            prefered_batch_size=[], 
                            no_of_instance=1, 
                            enable_gpu_optimization=False, 
                            precision_for_gpu_optimization="FP16",
                            max_batch_size=128):
        """
        Function helps in creation of configuration file required while 
        deploying model on tensorrt inference server.


        Parameters
        ----------
        saved_model_dir : string
            Name of directory of tensorflow saved model.

        data_type : {"TYPE_FP32", "TYPE_BOOL", "TYPE_UINT8", 
                    "TYPE_UINT16", "TYPE_UINT32", "TYPE_UINT64",
                    "TYPE_INT8", "TYPE_INT16", "TYPE_INT32", 
                    "TYPE_INT64", "TYPE_FP16", "TYPE_FP32", 
                    "TYPE_FP64", "TYPE_STRING"}, default: "TYPE_FP32"

        save_platform : string, default: "tensorflow_savedmodel"
            platform for which it should be saved.

        enable_dynamic_batching : bool, default: False
            parameter to enable dynamic batching while inferencing the model.

        prefered_batch_size : list, default=[]
            If dynamic batching is enabled what should be the preferences for batch size.

        no_of_instance : int, default: 1
            Represents the no of instance/copies of model you want to create for server.

        enable_gpu_optimization : bool, default=False
            If your deployment server is gpu enabled, then do you want to perform 
            gpu/platform specific optimizations or not.

        precision_for_gpu_optimization : {"FP16", "FP32", "INT8"}, default: "FP16"
            If you are enabling GPU optimization then what should be the precision value 
            used for optimization.
       
        """

        input_name, input_dim, output_name, output_dim=self.get_input_output_info(saved_model_dir)


        with open("config.pbtxt", "w") as file:
            file.write((self.content.format(saved_model_dir, 
                                            save_platform, 
                                            max_batch_size, 
                                            input_name, 
                                            data_type, 
                                            input_dim, 
                                            output_name, 
                                            data_type, 
                                            output_dim)))

        if(enable_dynamic_batching and prefered_batch_size):
            with open("config.pbtxt", "a") as file:
                file.write(self.prefered_batch_size.format(prefered_batch_size))

        elif(enable_dynamic_batching and (not prefered_batch_size)):
            with open("config.pbtxt", "a") as file:
                file.write(self.dynamic_batching)

        if(no_of_instance>1):
            with open("config.pbtxt", "a") as file:
                file.write(self.instance_group.format(no_of_instance))

        if(enable_gpu_optimization):
            with open("config.pbtxt", "a") as file:
                file.write(self.enable_gpu_optimization.format(precision_for_gpu_optimization))

    