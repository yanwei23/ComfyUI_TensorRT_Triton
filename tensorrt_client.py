#Put this in the custom_nodes folder, put your tensorrt engine files in ComfyUI/models/tensorrt/ (you will have to create the directory)

import torch
import os
import numpy as np

import comfy.model_base
import comfy.model_management
import comfy.model_patcher
import comfy.supported_models
import folder_paths

if "tensorrt" in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["tensorrt"][0].append(
        os.path.join(folder_paths.models_dir, "tensorrt"))
    folder_paths.folder_names_and_paths["tensorrt"][1].add(".engine")
else:
    folder_paths.folder_names_and_paths["tensorrt"] = (
        [os.path.join(folder_paths.models_dir, "tensorrt")], {".engine"})

import tensorrt as trt
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype

trt.init_libnvinfer_plugins(None, "")

logger = trt.Logger(trt.Logger.INFO)
runtime = trt.Runtime(logger)

class TrTUnet_Client:
    def __init__(self, server_addr, model_name):
        self._client = triton_client = httpclient.InferenceServerClient(
                url=server_addr,
                verbose=True,
            ) 
        self.model_name = model_name
        self.dtype = torch.float16 # useless
        self.engine_bs = 1

    def __call__(self, x, timesteps, context, y=None, control=None, transformer_options=None, **kwargs):
        internal_bs = x.shape[0]
        final_inputs = {}
        for i in range(internal_bs):
            index_slice = torch.IntTensor([i])
            final_inputs["x"] = x.cpu().index_select(0, index_slice)
            final_inputs["timesteps"] = timesteps.cpu().index_select(0, index_slice)
            final_inputs["context"] = context.cpu().index_select(0, index_slice)
            if y is not None:
                final_inputs["y"] = y.cpu().index_select(0, index_slice)

            for k,v in final_inputs.items():
                if isinstance(v, torch.Tensor):
                    final_inputs[k] = v.cpu().numpy()
                if final_inputs[k].dtype == np.float32:
                    final_inputs[k] = final_inputs[k].astype(np.float16)
    
            inputs = []
            outputs = []
            inputs.append(httpclient.InferInput("x", final_inputs["x"].shape, "FP16"))
            inputs.append(httpclient.InferInput("timesteps", final_inputs["timesteps"].shape, "FP16"))
            inputs.append(httpclient.InferInput("context", final_inputs["context"].shape, "FP16"))
            inputs.append(httpclient.InferInput("y", final_inputs["y"].shape, "FP16"))
    
            # Initialize the data
            inputs[0].set_data_from_numpy(final_inputs['x'], binary_data=True)
            inputs[1].set_data_from_numpy(final_inputs['timesteps'], binary_data=True)
            inputs[2].set_data_from_numpy(final_inputs['context'], binary_data=True)
            inputs[3].set_data_from_numpy(final_inputs['y'], binary_data=True)
    
            outputs.append(httpclient.InferRequestedOutput("h", binary_data=True))
    
            results = self._client.infer(
                self.model_name,
                inputs,
                outputs=outputs,
                query_params=None,
                headers=None,
                request_compression_algorithm=None,
                response_compression_algorithm=None
            )
        output = torch.from_numpy(results.as_numpy("h"))

        return output.to('cuda')

    def load_state_dict(self, sd, strict=False):
        pass

    def state_dict(self):
        return {}    

class TensorRTClient:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "server_address": ("STRING", {"default": "127.0.0.1:8000"}),
                    "model_type": ("STRING", {"default":"sdxl_base"}),
            }
        }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "get_infer_client"
    CATEGORY = "TensorRT"

    def get_infer_client(self, server_address, model_type):
        client = TrTUnet_Client(server_address, model_type)

        if model_type == "sdxl_base":
            conf = comfy.supported_models.SDXL({"adm_in_channels": 2816})
            conf.unet_config["disable_unet_model_creation"] = True
            model = comfy.model_base.SDXL(conf)
        elif model_type == "sdxl_refiner":
            conf = comfy.supported_models.SDXLRefiner(
                {"adm_in_channels": 2560})
            conf.unet_config["disable_unet_model_creation"] = True
            model = comfy.model_base.SDXLRefiner(conf)
        elif model_type == "sd1.x":
            conf = comfy.supported_models.SD15({})
            conf.unet_config["disable_unet_model_creation"] = True
            model = comfy.model_base.BaseModel(conf)
        elif model_type == "sd2.x-768v":
            conf = comfy.supported_models.SD20({})
            conf.unet_config["disable_unet_model_creation"] = True
            model = comfy.model_base.BaseModel(conf, model_type=comfy.model_base.ModelType.V_PREDICTION)
        elif model_type == "svd":
            conf = comfy.supported_models.SVD_img2vid({})
            conf.unet_config["disable_unet_model_creation"] = True
            model = conf.get_model({})
        elif model_type == "sd3":
            conf = comfy.supported_models.SD3({})
            conf.unet_config["disable_unet_model_creation"] = True
            model = conf.get_model({})
        elif model_type == "auraflow":
            conf = comfy.supported_models.AuraFlow({})
            conf.unet_config["disable_unet_model_creation"] = True
            model = conf.get_model({})
        elif model_type == "flux_dev":
            conf = comfy.supported_models.Flux({})
            conf.unet_config["disable_unet_model_creation"] = True
            model = conf.get_model({})
            unet.dtype = torch.bfloat16 #TODO: autodetect
        elif model_type == "flux_schnell":
            conf = comfy.supported_models.FluxSchnell({})
            conf.unet_config["disable_unet_model_creation"] = True
            model = conf.get_model({})
            unet.dtype = torch.bfloat16 #TODO: autodetect

        model.diffusion_model = client
        model.memory_required = lambda *args, **kwargs: 0

        return (comfy.model_patcher.ModelPatcher(model, load_device=comfy.model_management.get_torch_device(),
                                                 offload_device=comfy.model_management.unet_offload_device()),)

NODE_CLASS_MAPPINGS = {
    "TensorRTClient": TensorRTClient,
}
