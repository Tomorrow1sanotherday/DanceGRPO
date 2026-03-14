export WANDB_DISABLED=true
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online

# export HF_HOME=/mnt/ali-sh-1/usr/tusen/tmp-dev/shijian/.cache

# sudo apt-get update
# yes | sudo apt-get install python3-tk

# git clone https://github.com/tgxs002/HPSv2.git
# cd HPSv2
# pip install -e . 
# cd ..

# mkdir images_same

SD_MODEL_PATH="./data/stable-diffusion-v1-5"
REWARD_MODEL_NAME="llava-v1.6-13b"
TEMP_IMAGE_DIR="./temp_images"

# Checkpoint settings
CHECKPOINT_DIR="./checkpoints"
SAVE_FREQ=340

# Curriculum sampler settings: "timestep", "balance", "cosine", "gaussian"
CURRICULUM_STRATEGY="timestep"
CURRICULUM_TOTAL_STEPS=1000 #only for cosine and gaussian
CURRICULUM_ALPHA=1.0
CURRICULUM_BETA=1.0

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
