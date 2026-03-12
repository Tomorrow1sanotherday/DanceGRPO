#!/bin/bash
#先手动安装：pip install setuptools


# install torch
pip install psutil

pip install torch==2.5.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# install FA2 and diffusers
pip install packaging ninja && pip install flash-attn==2.7.0.post2 --no-build-isolation --no-cache-dir

# install lint requirements
pip install -r requirements-lint.txt

# install fastvideo (editable mode)
pip install -e .

# install extra deps
pip install ml-collections absl-py inflect==6.0.4 pydantic==1.10.9 huggingface_hub==0.24.0 protobuf==3.20.0 accelerate

# pip install setuptools

# pip install t2v-metrics

# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git

# pip install git+https://github.com/openai/CLIP.git

# pip install git+https://github.com/linzhiqiu/pytorchvideo.git

# pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# #conda install -c conda-forge ffmpeg=6.1