CUDA_VISIBLE_DEVICES=0 python train_all.py PRETRAIN --dataset PACS \
                    --deterministic --seed 0 --algorithm HTCL \
                    --generate_pattern --num_clusters 3 \
                    --pretrain --pretrain_epoch 40