import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import config as cfg

# --- Dual-Stream Encoder Components ---

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

class FactorizationMachine(nn.Module):
    def __init__(self, n_features, k_latent):
        super(FactorizationMachine, self).__init__()
        self.n = n_features
        self.k = k_latent
        self.linear = nn.Linear(self.n, 1, bias=True)
        self.v = nn.Parameter(torch.randn(self.n, self.k))
        nn.init.xavier_uniform_(self.v)

    def forward(self, x):
        # x is a batch of feature vectors
        linear_part = self.linear(x) # (batch, 1)
        
        inter_1 = torch.mm(x, self.v) # (batch, k)
        inter_2 = torch.mm(x.pow(2), self.v.pow(2)) # (batch, k)
        
        interaction_part = 0.5 * torch.sum(inter_1.pow(2) - inter_2, dim=1, keepdim=True) # (batch, 1)
        
        return linear_part + interaction_part

# --- SPERC Hierarchical Policy Networks ---

class DDPGActor(nn.Module):
    def __init__(self, state_dim, hidden_dim, max_action):
        super(DDPGActor, self).__init__()
        self.max_action = max_action
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        # Sigmoid to get action between 0 and 1, then scale
        return self.max_action * torch.sigmoid(self.l3(a))

class DuelingDQN(nn.Module):
    def __init__(self, state_dim, hidden_dim, n_actions):
        super(DuelingDQN, self).__init__()
        self.n_actions = n_actions

        # Advantage Stream
        self.adv_l1 = nn.Linear(state_dim, hidden_dim)
        self.adv_l2 = nn.Linear(hidden_dim, n_actions)

        # Value Stream
        self.val_l1 = nn.Linear(state_dim, hidden_dim)
        self.val_l2 = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        # state is expected to be [batch_size, n_actions, state_per_action_dim]
        # We process each action's state independently
        
        adv = F.relu(self.adv_l1(state))
        adv = self.adv_l2(adv) # [batch, n_actions, 1]

        val = F.relu(self.val_l1(state))
        val = self.val_l2(val) # [batch, n_actions, 1]
        
        # Average the value across all possible actions for a stable V(s) estimate
        avg_val = val.mean(dim=1, keepdim=True) # [batch, 1, 1]
        
        q_vals = avg_val + adv - adv.mean(dim=1, keepdim=True) # [batch, n_actions, 1]
        return q_vals.squeeze(-1) # [batch, n_actions]

# --- Baseline Models ---

class DDRLAgentModel(nn.Module):
    """A single DDPG agent that outputs both discrete and continuous actions."""
    def __init__(self, state_dim, n_nodes, max_cpu):
        super().__init__()
        self.max_cpu = max_cpu
        self.n_nodes = n_nodes
        
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        # Output N+1 values: N for node selection (logits), 1 for CPU allocation
        self.l3 = nn.Linear(256, n_nodes + 1)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        
        placement_logits = x[:, :self.n_nodes]
        cpu_output = self.max_cpu * torch.sigmoid(x[:, self.n_nodes:])
        
        return placement_logits, cpu_output