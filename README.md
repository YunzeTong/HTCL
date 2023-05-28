# Quantitatively Measuring and Contrastively Exploring Heterogeneity for Domain Generalization

Official PyTorch implementation of [Quantitatively Measuring and Contrastively Exploring Heterogeneity for Domain Generalization](https://arxiv.org/abs/2305.15889).


Note that this project is built upon [DomainBed@3fe9d7](https://github.com/facebookresearch/DomainBed/tree/3fe9d7bb4bc14777a42b3a9be8dd887e709ec414) and [SWAD](https://github.com/khanrc/swad).


## Preparation

### Dependencies

see `requirements.txt`

### Environments

Environment details used for our study.

```
Python: 3.7.15
PyTorch: 1.10.0+cu111
Torchvision: 0.11.0+cu111
CUDA: 12.0
NumPy: 1.21.6
```

## How to Run
To run the algorithm proposed in the paper completely, you should pretrain the feature extractor and classifier in advance for the first stage (heterogeneous dividing pattern generation). Take `PACS` as an example, you can run the following command:

```
CUDA_VISIBLE_DEVICES=0 python train_all.py PRETRAIN --dataset PACS --deterministic --seed 0 --algorithm HTCL --generate_pattern --num_clusters 3 --pretrain --pretrain_epoch 40
```

Then the pretrained feature extractor and classifier are stored in corresponding path, see [here](./heterolize/feature_heterolizer.py) for more details. Note that this process is completely independent of the main branch of DomainBed and each pretrained model won't see the corresponding test domain during training. In addition, the pretrained model won't involve in the final prediction. It is only used to help generate pattern.

After pretraining, you can just run another standard command to run the othe whole process following DomainBed:

```
CUDA_VISIBLE_DEVICES=0 python train_all.py HTCL --dataset PACS --deterministic --trial_seed 0 --algorithm HTCL --generate_pattern --num_clusters 3 --steps 5000 --checkpoint_freq 500 --data_dir /my/datasets/path
```


### Reproduce the results of the paper

Our method is two stages and the lack of the first stage (heterogeneous dividing pattern generation) won't lead the results reported in the paper. In other words, simple set `--algorithm` as HTCL is not enough.
Note that the difference in a detailed environment or uncontrolled randomness may bring a little different result from the paper.
Take `PACS` dataset as an example:

- PACS

```
CUDA_VISIBLE_DEVICES=0 python train_all.py HTCL --dataset PACS --deterministic --trial_seed 0 --algorithm HTCL --generate_pattern --num_clusters 3 --steps 5000 --checkpoint_freq 500 --data_dir /my/datasets/path
CUDA_VISIBLE_DEVICES=1 python train_all.py HTCL --dataset PACS --deterministic --trial_seed 1 --algorithm HTCL --generate_pattern --num_clusters 3 --steps 5000 --checkpoint_freq 500 --data_dir /my/datasets/path
CUDA_VISIBLE_DEVICES=2 python train_all.py HTCL --dataset PACS --deterministic --trial_seed 2 --algorithm HTCL --generate_pattern --num_clusters 3 --steps 5000 --checkpoint_freq 500 --data_dir /my/datasets/path
```

You can also use [run_pretrain.sh](./run_pretrain.sh) and [run.sh](./run.sh).

## License

This source code is released under the MIT license, included [here](./LICENSE).

This project includes some code from [DomainBed](https://github.com/facebookresearch/DomainBed/tree/3fe9d7bb4bc14777a42b3a9be8dd887e709ec414) and [SWAD](https://github.com/khanrc/swad), also MIT licensed.