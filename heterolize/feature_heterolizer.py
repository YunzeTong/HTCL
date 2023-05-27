import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
import os
import torchvision
import random
import numpy as np
import sys
sys.path.append('../')

from heterolize.dataset import EnvChangeDataset as EDataset

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

class AuxiliaryClassifier(nn.Module):
    """help to train featurizer, 如果使用CL可以考虑不使用本模块"""
    def __init__(self, feature_dim, num_labels):
        super(AuxiliaryClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, num_labels),
            # nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.network(x)

class Identity(nn.Module):
    """An identity layer"""

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class NormalizeFeaturizer(nn.Module):
    def __init__(self, input_shape, n_outputs):
        super(NormalizeFeaturizer, self).__init__()
        self.network = torchvision.models.resnet18(pretrained=True)
        self.n_outputs = n_outputs

        del self.network.fc
        # projection head
        self.network.fc = nn.Sequential(
            nn.Linear(512, n_outputs),
            nn.BatchNorm1d(n_outputs)
        )
        # if use_resnet50:
        #     self.network = torchvision.models.resnet50(pretrained=True)
        #     finish_dim = 2048
        # else:
        #     self.network = torchvision.models.resnet18(pretrained=True)
        #     finish_dim = 512
        
    def forward(self, x):
        return self.network(x)


class FeatureHeterolizer:
    """ 
    extract heterogeneous feature with given pure dataset (contain env_split)
    """
    def __init__(self, args, H_dataset, num_envs, logger):
        if args.dataset == "ColoredMNIST":
            input_shape = (2, 28, 28)
        else:
            input_shape = (3, 224, 224)
        self.featurizer = NormalizeFeaturizer(input_shape=input_shape, n_outputs=args.feature_dim).to(device)
        self.classifier = AuxiliaryClassifier(self.featurizer.n_outputs, H_dataset.num_labels).to(device)

        self.pretrain_batch_size = args.pretrain_batch_size
        self.pretrain_epoch = args.pretrain_epoch
        self.hf_epoch = args.hf_epoch

        self.num_envs = num_envs
        self.logger = logger

        self.H_dataset = H_dataset
    
    def pretrain(self, args, dataset:EDataset, dir):
        """use with AuxiliaryClassifier"""
        data_loader = DataLoader(dataset, batch_size=self.pretrain_batch_size, shuffle=True)
        pretrain_optimizer = optim.Adam(list(self.featurizer.parameters()) + list(self.classifier.parameters()), lr=1e-3, weight_decay=1e-6)
        self.logger.info("Start to pretrain...")
        self.featurizer.train()
        for epoch_i in range(self.pretrain_epoch):
            total_loss = 0
            for imgs, labels, index_in_dataset in data_loader:
                if imgs.shape[0] == 1:
                    continue 
                features = self.featurizer(imgs)
                label_prob = self.classifier(features)  
                loss = F.cross_entropy(label_prob, labels)
                pretrain_optimizer.zero_grad()
                loss.backward()
                pretrain_optimizer.step()
                total_loss += loss.data
            self.logger.info("     [Heterogeneity Exploration Pretrain] Epoch: {:d}, Loss: {:.3f}".format(epoch_i, total_loss))
            if (epoch_i + 1) % 20 == 0:
                os.makedirs(dir, exist_ok=True)
                featurizer_path = dir + f'/{args.seed}s_res18_{args.feature_dim}d_{epoch_i+1}e_featurizer.pt'
                classifier_path = dir + f'/{args.seed}s_res18_{args.feature_dim}d_{epoch_i+1}e_classifier.pt'
                torch.save(self.featurizer.state_dict(), featurizer_path)
                torch.save(self.classifier.state_dict(), classifier_path)

    def explore_heterogeneity(self, env_indicator, lambda_1=1e-3):
        """the process of exploring heterogeneity"""
        self.optimizer = optim.Adam(self.featurizer.parameters(), lr=1e-3, weight_decay=1e-6)
        # find all subgroups' indices
        total_indices = []
        for env_i in range(self.num_envs):
            total_indice = torch.nonzero(env_indicator == env_i).ravel()
            # assert total_indice.shape[0] != 0, f"STOP: One cluster has no samples, {torch.unique(env_indicator)}"
            total_indices.append(total_indice)

        self.logger.info(f"[Heterogeneity Exploration] Number of Domains: {len(total_indices)}; Each domain's contained number: {[total_indice.shape[0] for total_indice in total_indices]}")

        # set dataloader
        dataloader = DataLoader(self.H_dataset, batch_size=64, shuffle=True)
        
        # train featurizer only to enhance extracting
        self.featurizer.train()
        self.logger.info("Start to explore heterogeneity...")
        for epoch_i in range(self.hf_epoch):
            total_CL_loss = 0
            total_classify_loss = 0

            for idx, (imgs, labels, index_in_dataset) in enumerate(dataloader):
                if imgs.shape[0] == 1:
                    continue
                feature = self.featurizer(imgs)

                indices = []
                part_env_indicator = env_indicator[index_in_dataset]
                for env_i in range(self.num_envs):
                    indice = torch.nonzero(part_env_indicator == env_i).ravel()
                    if indice.shape[0] != 0:
                        indices.append(indice)
                
                CL_loss = 0
                if len(indices) != 1:
                    labels_own_in_batch = torch.unique(labels)
                    for label_idx in range(labels_own_in_batch.shape[0]):
                        label = labels_own_in_batch[label_idx]
                        same_label_indices = torch.nonzero(labels == label).ravel()
                        specific_f_list = self.split_by_env(same_label_indices, feature, part_env_indicator)
                        if len(specific_f_list) > 1:
                            for env_1 in range(len(specific_f_list) - 1):
                                for env_2 in range(env_1 + 1, len(specific_f_list)):
                                    f1, f2 = specific_f_list[env_1], specific_f_list[env_2]
                                    bigger_dist = self.dist(f1, f2)
                                    smaller_dist_1 = self.dist(f1, f1)
                                    smaller_dist_2 = self.dist(f2, f2)
                                    CL_loss += - torch.log(bigger_dist / (smaller_dist_1 + smaller_dist_2))
                CL_loss *= lambda_1

                # calculate classify loss
                pred = self.classifier(feature)
                classify_loss = F.cross_entropy(pred, labels)

                self.optimizer.zero_grad()
                total_epoch_loss = classify_loss + CL_loss
                total_epoch_loss.backward()
                self.optimizer.step()
                # if (idx + 1) % 40 == 0:
                #     print("Iter %d, classify loss: %.2f, CL loss: %.2f, cluster: %d" % (idx, classify_loss.data, CL_loss, len(indices)))
                
                if CL_loss != 0:
                    total_CL_loss += CL_loss.data
                total_classify_loss += classify_loss.data
            
            self.logger.info("  Epoch {:d}, Classify Loss: {:.3f}, Contrastive Loss: {:.3f}"\
                                                        .format(epoch_i, total_classify_loss, total_CL_loss))

        # return new low-dim feature
        self.featurizer.eval()
        distance = self.measure_heterogeneity(env_indicator)
        
        new_features = []
        imgs_dataloader = DataLoader(self.H_dataset, batch_size=32, shuffle=False) # originally use pure img dataset
        with torch.no_grad():
            for imgs, _, _ in imgs_dataloader:
                feature = self.featurizer(imgs)
                new_features.append(feature.to("cpu"))
        new_features = torch.cat(new_features, dim=0)
        return new_features.clone().detach(), distance

    def split_by_env(self, indices, total_features, env_indicator):
        """
        return:
        - a list whose every element is a tensor from same env with the same label
        """
        features_list = []
        features = total_features[indices]
        envs = env_indicator[indices]
        unique_envs = torch.unique(envs) # contain the specific env number 
        for idx in range(unique_envs.shape[0]):
            specific_env = unique_envs[idx]
            specific_env_indices = torch.nonzero(envs == specific_env).ravel()
            if specific_env_indices.shape[0] > 1:
                features_list.append(features[specific_env_indices])
        return features_list

    def dist(self, x, y, type="L2"):
        if type == "L2":
            def distance(t1, t2): # avoid self subtract cause sqrt 0
                return torch.sum(\
                            torch.sqrt(\
                                    torch.sum((t1-t2) ** 2, dim=1) + 1e-8 \
                                      )\
                                )
                # return torch.sqrt(torch.sum((t1-t2) ** 2))
        elif type == "L1":
            def distance(t1, t2):
                return torch.sum(torch.abs(t1-t2))
        total_distance = 0
        for idx_x in range(x.shape[0]):
            total_distance += distance(x[idx_x], y)
        if torch.equal(x, y):
            return total_distance / (x.shape[0] * (x.shape[0] - 1) / 2)
        else:
            return total_distance / x.shape[0] / y.shape[0]

    def measure_heterogeneity(self, env_indicator):
        # set dataloader
        dataloader = DataLoader(self.H_dataset, batch_size=64, shuffle=True)
        
        with torch.no_grad():
            # train featurizer only to enhance extracting
            self.featurizer.eval()
            
            total_distance = 0
            num_pairs_used = 0
            
            for idx, (imgs, labels, index_in_dataset) in enumerate(dataloader):
                
                feature = self.featurizer(imgs)
                part_env_indicator = env_indicator[index_in_dataset]

                if torch.unique(part_env_indicator).shape[0] == 1: # only have one env, continue
                    continue 
                
                distance_proportion = 0

                labels_own_in_batch = torch.unique(labels)
                for label_idx in range(labels_own_in_batch.shape[0]):
                    label = labels_own_in_batch[label_idx]
                    same_label_indices = torch.nonzero(labels == label).ravel() # 本loader中label为`label`的index
                    specific_f_list = self.split_by_env(same_label_indices, feature, part_env_indicator)
                    # print(f"[TEST DEBUG] label:{label}, {len(specific_f_list)}")
                    if len(specific_f_list) > 1:
                        for env_1 in range(len(specific_f_list) - 1):
                            for env_2 in range(env_1 + 1, len(specific_f_list)):
                                f1, f2 = specific_f_list[env_1], specific_f_list[env_2]
                                bigger_dist = self.dist(f1, f2)
                                smaller_dist_1 = self.dist(f1, f1)
                                smaller_dist_2 = self.dist(f2, f2)
                                distance_proportion += torch.log(bigger_dist / (smaller_dist_1 + smaller_dist_2))
                                num_pairs_used += 1
                
                total_distance += distance_proportion
        self.logger.info(f" [Heterogeneity Exploration] total distance:{total_distance}, num_pairs_used: {num_pairs_used}")
        if num_pairs_used == 0:
            return float('-inf')
        else:
            return total_distance / num_pairs_used