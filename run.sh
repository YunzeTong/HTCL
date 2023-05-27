CUDA_VISIBLE_DEVICES=4 python train_all.py HYP --dataset PACS \
                    --deterministic --trial_seed 0 --algorithm Don \
                    --generate_pattern --num_clusters 3 --main_epoch 5 --hf_epoch 5 --es_epoch 20 --pretrain_epoch 40 \
                    --steps 5000 --checkpoint_freq 500 --lambda_1 1 --test_envs 3;

# CUDA_VISIBLE_DEVICES=1 python train_all.py HYP --dataset PACS \
#                     --deterministic --trial_seed 1 --algorithm Don \
#                     --generate_pattern --num_clusters 3 --main_epoch 5 --hf_epoch 5 --es_epoch 20 --pretrain_epoch 40 \
#                     --steps 5000 --checkpoint_freq 500 --lambda_1 1;

# CUDA_VISIBLE_DEVICES=1 python train_all.py HYP --dataset PACS \
#                     --deterministic --trial_seed 0 --algorithm Don \
#                     --generate_pattern --num_clusters 3 --main_epoch 5 --hf_epoch 5 --es_epoch 20 --pretrain_epoch 40 \
#                     --steps 5000 --checkpoint_freq 500 --lambda_1 0;

# CUDA_VISIBLE_DEVICES=1 python train_all.py HYP --dataset PACS \
#                     --deterministic --trial_seed 1 --algorithm Don \
#                     --generate_pattern --num_clusters 3 --main_epoch 5 --hf_epoch 5 --es_epoch 20 --pretrain_epoch 40 \
#                     --steps 5000 --checkpoint_freq 500 --lambda_cont 2.5;