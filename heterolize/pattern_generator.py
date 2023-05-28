import sys
sys.path.append("../")
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from heterolize.utils import cross_entropy
from sklearn.cluster import KMeans

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

class MLP(nn.Module):
    """currently not used BN/dropout"""

    def __init__(self, n_inputs, n_outputs, num_layers=2, center_dim=256):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, center_dim)
        self.hiddens = nn.ModuleList(
            [
                nn.Linear(center_dim, center_dim)
                for _ in range(num_layers)
            ]
        )
        self.output = nn.Linear(center_dim, n_outputs)
        self.n_outputs = n_outputs
        nn.init.xavier_uniform_(self.input.weight)
        nn.init.xavier_uniform_(self.output.weight)
        for hidden in self.hiddens:
            nn.init.xavier_uniform_(hidden.weight)

    def forward(self, x):
        x = self.input(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = F.relu(x)
        x = self.output(x)
        return x
       

class PatternGenerator:
    def __init__(self, args, labels, num_labels, num_envs, feature_dim, logger):
        
        # not change during main training
        self.args = args
        self.num_labels = num_labels
        self.num_envs = num_envs
        self.labels = labels.to(device)
        self.feature_dim = feature_dim
        self.logger = logger
  

    def generate_group(self, features, mode=None):
        self.logger.info(" [Pattern Generation] Start to generate dividing pattern...")
        self.features = features.to(device)
        cluster_predicator = MLP(features.shape[1], self.num_envs, num_layers=4).to(device)          
        optimizer = optim.Adam(cluster_predicator.parameters(), lr=1e-3)
        # cluster by each kind of labels
        label_indices = []
        for i in range(self.num_labels):
            label_indice = torch.nonzero(self.labels == i).ravel()
            label_indices.append(label_indice)

        for epoch_i in range(self.args.es_epoch):
            clusters = cluster_predicator(self.features) # (n, num_env)
            clusters = F.softmax(clusters, dim=1)
            # self.avg_choose_prob(clusters)
            total_loss = 0
            CE_loss = 0
            penalty = 0
            penalty_min = 0

            for i in range(self.num_labels):
                label_indice = label_indices[i]
                cluster = clusters[label_indice]
                # self.avg_choose_prob(cluster, i)
                assert cluster.shape == (label_indice.shape[0], clusters.shape[1]), "dimension false"
                if mode == "avg":
                    cluster_avg = torch.mean(cluster, dim=0)
                    CE_loss -= cross_entropy(cluster_avg, 0)
                    penalty += 0.1 * (1/cluster_avg.shape[0] - torch.min(cluster_avg))
                else:
                    min_number_env = self.avg_choose_prob(cluster)
                    cluster_avg = torch.mean(cluster, dim=0)
                    CE_loss -= cross_entropy(cluster_avg, 0)
                    CE_loss -= torch.mean(cross_entropy(cluster)) 
                    penalty += (1/cluster_avg.shape[0] - torch.min(cluster_avg))
                    penalty_min += torch.relu(1/cluster_avg.shape[0] - torch.mean(cluster[:, min_number_env]))

            total_loss = CE_loss + penalty + penalty_min
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if (1 + epoch_i) % 5 == 0:
                self.logger.info("  Epoch {:d}; Loss: {:.3f}, CE: {:.3f}, penalty: {:3f}, penalty_min: {:3f}".\
                                                    format(epoch_i, total_loss.data, CE_loss.data, penalty.data, penalty_min.data))

        with torch.no_grad():
            final_env_indicator = cluster_predicator(self.features)
            final_env_indicator = torch.argmax(final_env_indicator, dim=1).ravel()
            # assert final_env_indicator.shape == (self.features.shape[0], 1), f"{final_env_indicator.shape}"
        self.logger.info(f" [Pattern Generation] Split into {self.num_envs} clusters | num: {[torch.nonzero(final_env_indicator==i).ravel().shape[0] for i in range(self.num_envs)]}")
        final_env_indicator = final_env_indicator.to("cpu")
        return final_env_indicator

    def avg_choose_prob(self, clusters):
        with torch.no_grad():
            prob, env_indices = torch.max(clusters, dim=1)
            own_envs = torch.unique(env_indices)
            avg_confidence_probs = torch.zeros(own_envs.shape[0])
            num = torch.zeros(own_envs.shape[0])
            for i in range(own_envs.shape[0]):
                own_env = own_envs[i]
                env_indice = torch.nonzero(env_indices == own_env).ravel()
                avg_confidence_prob = torch.mean(prob[env_indice])
                avg_confidence_probs[i] = avg_confidence_prob
                num[i] = prob[env_indice].shape[0]
            min_num_index = torch.argmin(num)
            assert min_num_index >= 0 and min_num_index < clusters.shape[1]
            return min_num_index
        
    def KMeans_split(self, features):
        model = KMeans(self.num_envs, random_state=self.args.trial_seed).fit(features.numpy())
        group = model.labels_
        env_indicator = torch.from_numpy(group)
        assert env_indicator.shape[0] == features.shape[0], "dim false in KMeans"
        return env_indicator