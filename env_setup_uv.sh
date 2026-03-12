#!/bin/bash
#先手动安装：uv pip install setuptools
#sudo apt install ffmpeg

# install torch
uv pip install torch==2.5.0 torchvision --index-url https://download.pytorch.org/whl/cu121

# install FA2 and diffusers
uv pip install packaging ninja
uv pip install flash-attn==2.7.0.post2 --no-build-isolation

# install lint requirements
uv pip install -r requirements-lint.txt

# install fastvideo (editable mode)
uv pip install -e .

# install extra deps
uv pip install ml-collections absl-py inflect==6.0.4 pydantic==1.10.9 huggingface_hub==0.24.0 protobuf==3.20.0 accelerate

uv pip install setuptools

uv pip install t2v-metrics

uv pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git

uv pip install git+https://github.com/openai/CLIP.git

uv pip install git+https://github.com/linzhiqiu/pytorchvideo.git

uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl