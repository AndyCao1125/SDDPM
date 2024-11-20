python ../main_SDDPM.py \
    --train \
    --dataset='mnist' \
    --beta_1=1e-4 --beta_T=0.02 \
    --img_ch=1 \
    --encoding='rate' \
    --img_size=32 --timestep=4 \
    --parallel=True --sample_step=0 \
    --total_steps=500001 \
    --logdir='./logs'