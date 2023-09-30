export WANDB_API_KEY='db0c63baeacf1eeb82545a162529728192b83020'
export WANDB_PROJECT='dqn_flap'

# export WANDB_MODE=disabled
timesteps=200000
bufferSize=20000

python dqn_flap.py \
    --exp-name test \
    --track \
    --wandb-project-name $WANDB_PROJECT \
    --capture-video \
    --save-model \
    --total-timesteps $timesteps \
    --buffer-size $bufferSize \
    --hf-entity polowitty \
    --learning-rate 1e-3 \
    --exploration-fraction 0.7 \
    --end-e 0.2 \
    --rounding 10 \
    --tau 0.9


# installation
# pip install flappy-bird-gymnasium
# pip3 install torch torchvision torchaudio
# pip install stable-baselines3[extra]
# pip install wandb
# pip install moviepy