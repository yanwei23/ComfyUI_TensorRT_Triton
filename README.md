# ComfyUI_TensorRT_Triton
Triton server搭建：
1.	根据https://github.com/comfyanonymous/ComfyUI_TensorRT的指导，将模型checkpoint文件转为TensorRT所需的点engine文件，默认路径ComfyUI/output/tensorrt/model_name/engine文件。
2.	建立triton使用的model_repository文件夹。参考https://github.com/yanwei23/ComfyUI_TensorRT_Triton/tree/main/model_repository
3.	将第1步生成的engine文件复制到model_repository/model_name/下，并重命名为model.plan。model_name可自行指定。
4.	model_repository/model_name/config.pbtxt大多数情况不需要修改，不同模型可以重复配置一份，有lora需要单独配置，参考（https://github.com/triton-inference-server/tutorials/blob/17331012af74eab68ad7c86d8a4ae494272ca4f7/Popular_Models_Guide/Llava1.5/model_repository/tensorrt_llm/config.pbtxt）
5.	从ngc上下载triton server镜像，例子使用的是nvcr.io/nvidia/tritonserver:25.01-py3。不同版本可能会有一些依赖兼容问题。
容器参考启动命令：docker run -itd --gpus all --net=host --ipc=host --shm-size=2g -v /data/weyan/:/myworkspace nvcr.io/nvidia/tritonserver:25.01-py3 /bin/bash
Tritonserver启动参考命令：tritonserver --model-repository /myworkspace/model_repository/ --model-control-mode explicit --load-model sdxl_base

ComfyUI Node：
将tensorrt_client.py放入custom_node对应的custom_nodes/ComfyUI_TensorRT文件夹下。__init__.py也覆盖ComfyUI_TensorRT或对应新增TensorRTClient的代码。
这里加入了新的TensorRT Client节点，替换原TensorRT工作流中TensorRT_Loader。指定服务端地址和模型类型/名称（此处名称与模型的model_repository下的名称匹配）。

输出指向k-sampler。
即可执行。参考工作流为create_sdxl_tensorrt_client.json。


