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

torchrun --nproc_per_node=3 --master_port 19001 \
fastvideo/train_grpo_sd_curr.py --config fastvideo/config_sd/dgx.py:hps
