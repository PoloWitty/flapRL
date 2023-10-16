export WANDB_API_KEY='db0c63baeacf1eeb82545a162529728192b83020'
export WANDB_PROJECT='dqn_flap_20231011'

# export WANDB_MODE=disabled
timesteps=1000000
bufferSize=100000

python dqn_flap.py \
    --exp-name midDoubleDQN_bot_continualSpace \
    --track \
    --wandb-project-name $WANDB_PROJECT \
    --capture-video \
    --save-model \
    --total-timesteps $timesteps \
    --buffer-size $bufferSize \
    --hf-entity polowitty \
    --learning-rate 1e-3 \
    --exploration-fraction 0.5 \
    --end-e 0 \
    --start-e 0.1 \
    --action-downsample-ratio 1

# installation
# pip install flappy-bird-gymnasium
# pip3 install torch torchvision torchaudio
# pip install stable-baselines3[extra]
# pip install wandb
# pip install moviepy

# wandb sweep
# wandb sweep --project dqn_flap sweep.yaml
# wandb agent polowitty/dqn_flap/jjgyime5