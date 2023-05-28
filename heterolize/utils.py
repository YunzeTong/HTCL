import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from datetime import datetime
import copy

def store_imgs(args, imgs, labels, domain_labels, test_env):
    # build root folder
    root_dir = f"../mydataset/{args.dataset}_{args.test_env_name}_{args.num_clusters}"
    os.makedirs(root_dir, exist_ok=True)
    # build source env
    for env_i in range(args.num_clusters):
        env_dir = root_dir + f"/env_{env_i}"
        os.makedirs(env_dir, exist_ok=True)
        select_indices = torch.nonzero(domain_labels == env_i).ravel()  # tensor([1,3,5])
        for i in range(select_indices.shape[0]):
            index = select_indices[i] # index_th img should be in env_i
            # create folder for class
            label_dir = env_dir + f"/{labels[index]}"
            os.makedirs(label_dir, exist_ok=True)
            # store img
            imgs[index].save(label_dir + f"/{i}.jpg")

    # build target env
    folder = root_dir + '/target'
    os.makedirs(folder, exist_ok=True)
    for idx, (img, class_) in enumerate(test_env):
        label_folder = folder + '/' +str(class_)
        if not os.path.exists(label_folder):
            os.makedirs(label_folder, exist_ok=True)
        img.save(label_folder + '/' + str(idx) + '.jpg') 

def timestamp(fmt="%y%m%d_%H-%M-%S"):
    return datetime.now().strftime(fmt)

class Logger:
    def __init__(self, args, combine=True):
        if combine:
            log_dir = './heterolize/record/{}/{}'.format(args.dataset, args.test_env_name)
            log_file = log_dir + '/C_{}_{}_{}_{}_{}.txt'.format(\
                args.main_epoch, args.hf_epoch, args.es_epoch, args.num_clusters, timestamp()
            )
        else:
            log_dir = './record/{}/{}'.format(args.dataset, args.test_env_name)
            log_file = log_dir + '/S_{}_{}_{}_{}_{}.txt'.format(\
                args.main_epoch, args.hf_epoch, args.es_epoch, args.num_clusters, timestamp()
            )
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.log_file = log_file

    def info(self, print_str, print_=True):
        if print_:
            print(print_str)
        with open(self.log_file, 'a') as f:
            f.write(print_str)
            f.write('\n')

def cross_entropy(prob, dim=1):
    return -(prob * torch.log(prob)).sum(dim=dim)

class HeterogeneousDomain(data.Dataset):
    def __init__(self, E_dataset, select_indices, batch_size_needed):
        super(HeterogeneousDomain).__init__()
        # self.imgs, self.labels = [], []
        self.batch_size_needed = batch_size_needed

        self.own_indices = select_indices
        self.total_envs_dataset = E_dataset

    def __getitem__(self, index):
        actual_index = self.own_indices[index % self.own_indices.shape[0]]
        img, label, _ = self.total_envs_dataset[actual_index]
        return img, label
        # return self.imgs[index % len(self.labels)], self.labels[index % len(self.labels)]

    def __len__(self):
        return max(self.own_indices.shape[0], self.batch_size_needed)


class HeterogeneousDomains:
    def __init__(self, domains, n_clusters):
        
        self.domains = domains
        self.domain_nmb = n_clusters + 1 # denote how many domains(include test domain) 

    def __getitem__(self, index):
        return self.domains[index]

    def __len__(self):
        return len(self.domains)

    def add_params(self, num_classes, N_STEPS, CHECKPOINT_FREQ, N_WORKERS, input_shape):
        self.num_classes = num_classes
        self.N_STEPS = N_STEPS
        self.CHECKPOINT_FREQ = CHECKPOINT_FREQ
        self.N_WORKERS = N_WORKERS
        self.input_shape = input_shape
        self.environments = []
        for i in range(self.domain_nmb):
            if i == self.domain_nmb - 1:
                self.environments.append("test_domain")
            else:
                self.environments.append(f"domain_train_{i}")

def make_final_dataset(args, E_dataset, domain_labels, test_env, hparams=None):
    
    args.test_env = args.num_clusters      # always put target domain in the end with `args.num_clusters`; 
                                            # 0 to args.num_clusters - 1 are all train domains
    # if number of envs decreases
    own_cluster_idx = torch.unique(domain_labels)
    if own_cluster_idx.shape[0] != args.num_clusters:
        args.test_env = args.num_clusters - 1
    
    domains = []
    E_dataset.in_heterogeneity_exploration = False
    # last num_clusters are all train domains, final is target domain
    for env_i in range(args.num_clusters):
        select_indices = torch.nonzero(domain_labels == env_i).ravel()
        if select_indices.shape[0] == 0: # if a domain doesn't contain samples, just drop
            continue
        single_domain = HeterogeneousDomain(E_dataset, select_indices, int(hparams["batch_size"] / (1-args.holdout_fraction)))
        # for i in range(select_indices.shape[0]):
        #     index = select_indices[i] # index_th img should be in env_i
        #     single_domain.add_item(imgs[index], labels[index])
        domains.append(single_domain)
    # build target env
    domains.append(test_env)
    
    args.test_env = len(domains) - 1
    args.num_clusters = len(domains) - 1

    dataset = HeterogeneousDomains(domains, args.num_clusters)
    return dataset
