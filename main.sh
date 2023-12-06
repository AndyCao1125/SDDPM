CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main_SDDPM.py \
    --train \
    --dataset='cifar10' \
    --beta_1=1e-4 --beta_T=0.02 \
    --img_size=32 --timestep=4 --img_ch=3 \
    --parallel=True --sample_step=0 \
    --total_steps=500001 \
    --logdir='./logs'