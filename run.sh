CUDA_VISIBLE_DEVICES=4 python train_all.py HTCL --dataset PACS \
                    --deterministic --trial_seed 0 --algorithm HTCL \
                    --generate_pattern --num_clusters 3 --hf_epoch 5 --es_epoch 20 \
                    --steps 5000 --checkpoint_freq 500;

