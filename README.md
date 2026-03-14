## DanceGRPO SD GRPO 训练说明（简版）

**本仓库当前主要使用 `finetune_sd_grpo.sh` 进行 Stable Diffusion 的 GRPO 训练。下面是从环境搭建到启动训练的完整流程，按顺序执行即可。**

---

## 1. 环境准备

- **Python**: 3.10  
- **虚拟环境工具**: `uv`（建议已全局安装）  
- **GPU**: 需支持对应 CUDA 版本（默认使用 `torch==2.5.0` + `cu121` 源）  

### 1.1 创建并激活 uv 环境

这里以创建名为 `dancegrpo` 的虚拟环境为例（命令按你实际习惯来，关键是：环境名 `dancegrpo`，Python 版本 3.10）：

```bash
# 示例（按需替换为你自己的命令）
uv venv --python 3.10 dancegrpo

# 激活环境
source dancegrpo/bin/activate
```

确保此时：

- `python --version` 为 3.10.x  
- 已经处在名为 `dancegrpo` 的环境中  

---

## 2. 安装依赖（env_setup_uv.sh）

激活 `dancegrpo` 环境后，在项目根目录 `DanceGRPO` 下执行安装脚本。

### 2.1 切换到项目根目录

```bash
cd DanceGRPO
```

### 2.2 执行环境安装脚本

```bash
bash env_setup_uv.sh
```

`env_setup_uv.sh` 中主要做了以下几件事（使用 `uv pip`）：

- 安装基础依赖：`psutil`、`setuptools`、`packaging`、`ninja` 等  
- 安装 PyTorch：`torch==2.5.0`、`torchvision`（`cu121` 源）  
- 安装 `flash-attn` 及其额外 wheel  
- 安装 `requirements-lint.txt` 里的检查/开发依赖  
- 以可编辑模式安装本仓库：`uv pip install -e .`  
- 安装额外依赖：`ml-collections`、`absl-py`、`inflect==6.0.4`、`pydantic==1.10.9`、`huggingface_hub==0.24.0`、`protobuf==3.20.0`、`accelerate` 等  
- 安装评测与外部项目：`t2v-metrics`、`LLaVA-NeXT`、`openai/CLIP`、自定义 `pytorchvideo`  

脚本注释中还提示：

```bash
# 先手动安装：uv pip install setuptools
# sudo apt install ffmpeg
```

如有需要，先手动安装：

```bash
uv pip install setuptools
sudo apt install ffmpeg
```

---

## 3. 数据与模型准备

`scripts/finetune/finetune_sd_grpo.sh` 中相关默认路径为：

- `SD_MODEL_PATH="./data/stable-diffusion-v1-5"`  
- `REWARD_MODEL_NAME="llava-v1.6-13b"`  
- `TEMP_IMAGE_DIR="./temp_images"`  
- `CHECKPOINT_DIR="./checkpoints"`  

你需要根据这些变量准备好：

- 将 Stable Diffusion v1.5 模型权重放在 `./data/stable-diffusion-v1-5`  
- 确保能加载名为 `llava-v1.6-13b` 的奖励模型（例如从 Hugging Face 或本地路径）  
- 创建临时图片目录和 checkpoint 目录：

```bash
mkdir -p data/stable-diffusion-v1-5
mkdir -p temp_images
mkdir -p checkpoints
```

---

## 4. 训练脚本（finetune_sd_grpo.sh）

主要逻辑位于 `scripts/finetune/finetune_sd_grpo.sh`，核心包括：

- 设置 W&B 环境变量（`WANDB_DISABLED=true` 等）  
- 设置 SD 模型路径、奖励模型名称、临时图片目录、checkpoint 保存目录  
- 设置 curriculum 采样策略：
  - `CURRICULUM_STRATEGY="timestep"`（可选 `"balance"`, `"cosine"`, `"gaussian"` 等）
  - `CURRICULUM_TOTAL_STEPS`、`CURRICULUM_ALPHA`、`CURRICULUM_BETA` 控制曲线形状  
- 使用 `torchrun` 启动多卡训练：

```bash
torchrun --nproc_per_node=8 --master_port 19001 \
  fastvideo/train_grpo_sd_curr.py \
  --config fastvideo/config_sd/dgx.py:hps \
  --config.pretrained.model="${SD_MODEL_PATH}" \
  --config.reward_model_name="${REWARD_MODEL_NAME}" \
  --config.temp_image_dir="${TEMP_IMAGE_DIR}" \
  --config.checkpoint_dir="${CHECKPOINT_DIR}" \
  --config.save_freq="${SAVE_FREQ}" \
  --config.curriculum.strategy="${CURRICULUM_STRATEGY}" \
  --config.curriculum.total_steps="${CURRICULUM_TOTAL_STEPS}" \
  --config.curriculum.alpha="${CURRICULUM_ALPHA}" \
  --config.curriculum.beta="${CURRICULUM_BETA}"
```

- `--nproc_per_node=8` 表示使用 8 卡，如果你机器 GPU 数量不同，可以改成对应的卡数。  

---

## 5. 启动训练（一步到位）

假设你已经：

- 安装好 `uv`，并创建+激活了 `dancegrpo`（Python 3.10）环境  
- 在 `/root/autodl-tmp/DanceGRPO` 下跑完了 `bash env_setup_uv.sh`  
- 准备好了 SD 模型和奖励模型，对应目录已经就绪  

那么只需要：

```bash
cd /root/autodl-tmp/DanceGRPO
bash scripts/finetune/finetune_sd_grpo.sh
```

如果你把脚本拷到根目录并命名为 `finetune_sd_grpo.sh`，也可以：

```bash
cd /root/autodl-tmp/DanceGRPO
bash finetune_sd_grpo.sh
```

训练过程中：

- checkpoint 会保存到 `./checkpoints`  
- 临时图片写到 `./temp_images`  
- 如配置了 W&B，可在网页端查看日志和曲线  

---

## 6. 常用可调参数

你可以直接在 `scripts/finetune/finetune_sd_grpo.sh` 中改以下变量：

- **基础模型路径**：`SD_MODEL_PATH`  
- **奖励模型名称**：`REWARD_MODEL_NAME`  
- **Checkpoint 保存间隔**：`SAVE_FREQ`  
- **Curriculum 策略与参数**：`CURRICULUM_STRATEGY`、`CURRICULUM_TOTAL_STEPS`、`CURRICULUM_ALPHA`、`CURRICULUM_BETA`  
- **GPU 数量**：`torchrun --nproc_per_node=...`  

改完重新 `bash` 该脚本即可生效。

---

## 7. 使用细粒度 VQAScore 的注意事项

如果使用细粒度的 `vqa_score` 作为奖励信号，需要修改 `t2v_metrics` 包中 LLaVA 1.6 模型的默认问题模板，否则评分不够精准。

找到安装环境中的文件：

```
<your_env>/lib/python3.10/site-packages/t2v_metrics/models/vqascore_models/llava16_model.py
```

将第 11 行的：

```python
default_question_template = 'Does this figure show "{}"? Please answer yes or no.'
```

改为：

```python
default_question_template = '{} Answer only "yes" or "no", one word only.'
```

> **原因**：默认模板是粗粒度的二分类提问（"Does this figure show ..."），对复杂场景描述的区分度不够。改为直接将场景描述作为问题、并限制只回答 yes/no 一个词，可以获得更细粒度、更准确的 VQAScore。

---

## 8. 快速总结

1. 安装并使用 `uv` 创建 **Python 3.10** 环境 `dancegrpo`，激活；  
2. `cd /root/autodl-tmp/DanceGRPO`；  
3. 运行 `bash env_setup_uv.sh` 安装依赖；  
4. 准备好 `./data/stable-diffusion-v1-5` 模型权重等资源；  
5. 运行训练：`bash scripts/finetune/finetune_sd_grpo.sh`（或你自己放在根目录的 `bash finetune_sd_grpo.sh`）。  

按以上顺序执行，即可完成环境搭建并启动 DanceGRPO 的 Stable Diffusion GRPO 训练。
