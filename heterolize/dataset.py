import sys
sys.path.append("../")
import numpy as np
import torch
import torch.utils.data as data
from domainbed.datasets import transforms as DBT
from PIL.Image import Image
from heterolize.baseline.utils.IPIRM_utils import train_transform

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

class EnvChangeDataset(data.Dataset):
    def __init__(self, num_envs, num_labels):
        super(EnvChangeDataset).__init__()

        self.num_envs = num_envs
        self.num_labels = num_labels
        
        # self.imgs = []
        self.labels = [] 
        self.original_env = []
        self.transform = DBT.basic
        self.perform_tensor_transform = True

        self.old_envs = [] # every term is the old training domain from given dataset
        self.old_envs_num = []
        self.old2new_index_list = None

        # change during train
        self.domain_labels = []
        
    def __getitem__(self, index):
        old_env_idx, idx_in_old_env = self.find_old_env_index(index)   
        if self.perform_tensor_transform:
            # return self.transform(self.imgs[index]).to(device), self.labels[index].to(device), index
            return self.transform(self.old_envs[old_env_idx][idx_in_old_env][0]).to(device), \
                self.labels[index].to(device), index
        else:
            return self.old_envs[old_env_idx][idx_in_old_env][0], \
                self.labels[index], index

    def __len__(self):
        return len(self.old2new_index_list)

    def add_item(self, label, env):
        # self.imgs.append(img)
        self.labels.append(label)
        self.original_env.append(env)
        self.domain_labels.append(env)

    def shuffle(self, seed):
        new_index_list = [i for i in range(len(self.original_env))]
        np.random.RandomState(seed).shuffle(new_index_list)
        # self.imgs = [img for _, img in sorted(zip(new_index_list, self.imgs))]
        self.labels = [label for _, label in sorted(zip(new_index_list, self.labels))]
        self.original_env = [env for _, env in sorted(zip(new_index_list, self.original_env))]
        self.domain_labels = [env for _, env in sorted(zip(new_index_list, self.domain_labels))]

        self.old2new_index_list = new_index_list

    def find_old_env_index(self, index):
        """
        return old_env_idx, idx_in_old_env"""
        original_pos = self.old2new_index_list.index(index) # 在顺序排列的old env中所处位置
        old_env_idx = len(self.old_envs)
        idx_in_old_env = -1
        for env_idx, old_env_num in enumerate(self.old_envs_num):
            if original_pos < old_env_num:
                old_env_idx = env_idx
                idx_in_old_env = original_pos
                break
            else:
                original_pos -= old_env_num
        return old_env_idx, idx_in_old_env

    def update_domain_labels(self, new_domain_labels=None):
        if type(self.labels) == list:
            self.labels = torch.tensor(self.labels)
        if new_domain_labels == None:
            self.domain_labels = torch.tensor(self.domain_labels)
            # if self.num_envs < torch.unique(self.domain_labels).shape[0]:# TODO: under check
                # print(f"{self.num_envs} < {torch.unique(self.domain_labels)}")
            self.domain_labels = torch.randint(0, self.num_envs, self.domain_labels.shape) # randomly set domain labels for initialization
        else:
            self.domain_labels = new_domain_labels

    
