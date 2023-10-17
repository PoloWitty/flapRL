# wait for previous run to finish
# pid=17828
# tail --pid=$pid -f /dev/null

export WANDB_API_KEY='db0c63baeacf1eeb82545a162529728192b83020'
export WANDB_PROJECT='dqn_flap_20231011'

# export WANDB_MODE=disabled
timesteps=1000000
bufferSize=100000

python dqn_flap.py \
    --exp-name hardDoubleDQN_bot \
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
    --start-e 0.01 \
    --action-downsample-ratio 1 \
    --from-pretrained runs/FlappyBird-v0__midDoubleDQN_bot__1__1697438658/midDoubleDQN_bot.model

# # in case not run successfully, run reserve program
# conda activate base
# python ~/reserve/reserve.py

# installation
# pip install flappy-bird-gymnasium
# pip3 install torch torchvision torchaudio
# pip install stable-baselines3[extra]
# pip install wandb
# pip install moviepy

# wandb sweep
# wandb sweep --project dqn_flap sweep.yaml
# wandb agent polowitty/dqn_flap/jjgyime5