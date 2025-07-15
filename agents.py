import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import networkx as nx

import config as cfg
from models import GCN, FactorizationMachine, DDPGActor, DuelingDQN

# --- Replay Buffer ---
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# --- Base Agent Class ---
class BaseAgent:
    def select_action(self, state):
        raise NotImplementedError
    def update(self):
        pass
    def store_transition(self, *args):
        pass

# --- SPERC Agent ---
class SPERCAgent(BaseAgent):
    def __init__(self, env_params):
        self.p = env_params
        
        # SPERC specific architecture
        # Dual-Stream Encoder
        self.gcn = GCN(in_channels=2, hidden_channels=cfg.GCN_HIDDEN_LAYERS[0], out_channels=cfg.GCN_HIDDEN_LAYERS[1]).to(cfg.DEVICE)
        self.fm_input_dim = cfg.N_UNIQUE_LAYERS + cfg.N_UNIQUE_LAYERS
        self.fm = FactorizationMachine(self.fm_input_dim, cfg.FM_LATENT_DIM).to(cfg.DEVICE)
        
        # Hierarchical Policy
        # Combined state dim for one node: GCN_out + FM_out + service_features
        service_feature_dim = 1 # Just workload for simplicity
        self.state_per_node_dim = cfg.GCN_HIDDEN_LAYERS[1] + 1 + service_feature_dim
        
        self.ddpg_actor = DDPGActor(self.state_per_node_dim, cfg.DDPG_ACTOR_HIDDEN_SIZE, max(cfg.CPU_CAPACITY_RANGE)).to(cfg.DEVICE)
        self.dueling_dqn = DuelingDQN(self.state_per_node_dim + 1, cfg.DUELING_DQN_HIDDEN_SIZE, cfg.N_NODES).to(cfg.DEVICE)

        # Target Networks
        self.ddpg_actor_target = DDPGActor(self.state_per_node_dim, cfg.DDPG_ACTOR_HIDDEN_SIZE, max(cfg.CPU_CAPACITY_RANGE)).to(cfg.DEVICE)
        self.dueling_dqn_target = DuelingDQN(self.state_per_node_dim + 1, cfg.DUELING_DQN_HIDDEN_SIZE, cfg.N_NODES).to(cfg.DEVICE)
        self.ddpg_actor_target.load_state_dict(self.ddpg_actor.state_dict())
        self.dueling_dqn_target.load_state_dict(self.dueling_dqn.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.ddpg_actor.parameters(), lr=cfg.LEARNING_RATE)
        self.dqn_optimizer = optim.Adam(self.dueling_dqn.parameters(), lr=cfg.LEARNING_RATE)
        
        self.replay_buffer = ReplayBuffer(cfg.REPLAY_BUFFER_SIZE)
        self.steps_done = 0

    def _get_state_embedding(self, state):
        # 1. GCN Stream
        node_features = torch.FloatTensor(state['node_features']).to(cfg.DEVICE)
        edge_index = torch.LongTensor(list(nx.to_edgelist(state['graph']))).t().contiguous()[:, :2].to(cfg.DEVICE)
        gcn_embeds = self.gcn(node_features, edge_index)

        # 2. FM Stream
        service_layers = torch.FloatTensor(state['service']['r_q']).unsqueeze(0).to(cfg.DEVICE)
        node_layers = torch.FloatTensor(state['layer_matrix']).to(cfg.DEVICE)
        fm_embeds = []
        for i in range(cfg.N_NODES):
            fm_input = torch.cat([service_layers, node_layers[i].unsqueeze(0)], dim=1)
            fm_embeds.append(self.fm(fm_input))
        fm_embeds = torch.cat(fm_embeds, dim=0)

        # 3. Combine
        service_workload = torch.FloatTensor([state['service']['workload']]).unsqueeze(0).repeat(cfg.N_NODES, 1).to(cfg.DEVICE)
        
        full_embeds = torch.cat([gcn_embeds, fm_embeds, service_workload], dim=1)
        return full_embeds # [N_NODES, combined_feature_dim]

    def select_action(self, state, exploration=True):
        eps_threshold = 0.1 if exploration else 0.0 # simple exploration for eval
        if random.random() > eps_threshold:
            with torch.no_grad():
                state_embeds = self._get_state_embedding(state) # [N, state_per_node]
                
                # Low-level DDPG proposes continuous actions for all discrete choices
                proposed_cpus = self.ddpg_actor(state_embeds) # [N, 1]
                
                # High-level Dueling DQN evaluates each discrete choice
                dqn_input = torch.cat([state_embeds, proposed_cpus], dim=1)
                q_values = self.dueling_dqn(dqn_input.unsqueeze(0)) # [1, N]
                
                node_idx = q_values.argmax().item()
                cpu_alloc = proposed_cpus[node_idx].item()
        else:
            node_idx = random.randrange(cfg.N_NODES)
            cpu_alloc = np.random.uniform(0, state['node_features'][node_idx, 0])

        return (node_idx, cpu_alloc)

    def store_transition(self, *args):
        self.replay_buffer.push(*args)

     def _get_batch_embeddings(self, states_batch):
        """Helper to process a batch of states and return embeddings."""
        batch_size = len(states_batch)
        # We process each state in the batch individually since GCN and FM are not easily batched with variable graphs/inputs here.
        # In a production system, you'd use a library like PyG's Batch object for more efficient batching.
        
        all_state_embeds = []
        for state in states_batch:
            if state is not None:
                # Get embeddings for all nodes for this single state
                state_embeds = self._get_state_embedding(state) # [N_NODES, state_per_node_dim]
                all_state_embeds.append(state_embeds)
            else:
                # Handle terminal states (next_state is None)
                dummy_embeds = torch.zeros((cfg.N_NODES, self.state_per_node_dim), device=cfg.DEVICE)
                all_state_embeds.append(dummy_embeds)
                
        # Stack into a single batch tensor: [B, N_NODES, state_per_node_dim]
        return torch.stack(all_state_embeds)

    def update(self):
        if len(self.replay_buffer) < cfg.BATCH_SIZE:
            return

        transitions = self.replay_buffer.sample(cfg.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # --- Prepare batch data ---
        # `batch.state` and `batch.next_state` are tuples of dicts.
        # `batch.action` is a tuple of (node_idx, cpu_alloc) tuples.
        
        # Filter out transitions where next_state is None
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=cfg.DEVICE, dtype=torch.bool)
        non_final_next_states = [s for s in batch.next_state if s is not None]

        # Extract actions and rewards
        action_batch = np.array(batch.action)
        node_indices = torch.LongTensor(action_batch[:, 0]).unsqueeze(1).to(cfg.DEVICE) # [B, 1]
        cpu_allocs = torch.FloatTensor(action_batch[:, 1]).unsqueeze(1).to(cfg.DEVICE) # [B, 1]
        
        reward_batch = torch.FloatTensor(batch.reward).to(cfg.DEVICE) # [B]

        # Get embeddings for current states
        # [B, N_NODES, state_per_node_dim]
        current_state_embeds = self._get_batch_embeddings(batch.state)

        # --- 1. Calculate Target Q-values (for DQN update) ---
        next_q_values = torch.zeros(cfg.BATCH_SIZE, device=cfg.DEVICE)
        
        if len(non_final_next_states) > 0:
            # Get embeddings for non-final next states
            next_state_embeds = self._get_batch_embeddings(non_final_next_states) # [n_non_final, N_NODES, state_per_node_dim]
            
            with torch.no_grad():
                # Target DDPG Actor proposes next continuous actions
                next_proposed_cpus = self.ddpg_actor_target(next_state_embeds) # [n_non_final, N_NODES, 1]
                
                # Target Dueling DQN evaluates the next state-action pairs
                next_dqn_input = torch.cat([next_state_embeds, next_proposed_cpus], dim=-1)
                # The DuelingDQN expects [B, N_actions, features], which matches our shape.
                q_values_next = self.dueling_dqn_target(next_dqn_input) # [n_non_final, N_NODES]
                
                # We take the max Q-value over all possible placement actions in the next state
                max_next_q_values = q_values_next.max(1)[0]
                next_q_values[non_final_mask] = max_next_q_values

        # Compute the final target `y`
        # y = r + gamma * Q_target(s', a'*) for non-terminal states
        # y = r for terminal states
        y_target = reward_batch + (cfg.DISCOUNT_FACTOR * next_q_values)


        # --- 2. Update Dueling DQN (Critic for placement) ---
        self.dqn_optimizer.zero_grad()
        
        # The critic needs to evaluate the Q-value for the (state, action) pair from the buffer.
        # Since our DQN evaluates all actions at once, we compute them all and then select the one taken.
        
        # We need to use the actual CPU allocation from the batch for the chosen node
        # to correctly evaluate the Q(s,a) that led to the reward.
        # This requires a bit more care than a standard DQN.
        # For simplicity and to follow the hierarchical logic, let's evaluate Q(s, n, μ(s,n))
        # where n is the action from the buffer.
        
        # Re-compute proposed CPUs for current states using MAIN actor
        proposed_cpus_current = self.ddpg_actor(current_state_embeds) # [B, N_NODES, 1]
        
        # Create input for the DQN
        dqn_input_current = torch.cat([current_state_embeds, proposed_cpus_current], dim=-1)
        q_values_current = self.dueling_dqn(dqn_input_current) # [B, N_NODES]
        
        # Gather the Q-values corresponding to the actions (node_indices) that were actually taken
        q_value_for_action_taken = q_values_current.gather(1, node_indices).squeeze(1) # [B]
        
        # Calculate DQN loss
        dqn_loss = F.mse_loss(q_value_for_action_taken, y_target.detach())
        
        dqn_loss.backward()
        self.dqn_optimizer.step()
        
        # --- 3. Update DDPG Actor (Policy for allocation) ---
        self.actor_optimizer.zero_grad()

        # The actor's goal is to output actions that maximize the critic's (DQN's) Q-value.
        # The loss is the negative mean of the Q-values. We perform gradient ASCENT.
        
        # We need to re-compute proposed actions and evaluate them with the DQN,
        # but this time we let gradients flow back to the actor.
        proposed_cpus_for_actor_loss = self.ddpg_actor(current_state_embeds) # [B, N_NODES, 1]
        
        # The DDPG actor's weights should not be updated by the DQN's loss calculation,
        # so we re-use the current state embeddings but build a new graph for backprop.
        dqn_input_for_actor = torch.cat([current_state_embeds.detach(), proposed_cpus_for_actor_loss], dim=-1)
        
        actor_q_values = self.dueling_dqn(dqn_input_for_actor)
        
        # Actor loss is the negative mean of the Q-values. Maximizing Q is minimizing -Q.
        actor_loss = -actor_q_values.mean()
        
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- 4. Soft update target networks ---
        with torch.no_grad():
            for target_param, param in zip(self.dueling_dqn_target.parameters(), self.dueling_dqn.parameters()):
                target_param.data.copy_(cfg.TAU_SOFT_UPDATE * param.data + (1.0 - cfg.TAU_SOFT_UPDATE) * target_param.data)
            
            for target_param, param in zip(self.ddpg_actor_target.parameters(), self.ddpg_actor.parameters()):
                target_param.data.copy_(cfg.TAU_SOFT_UPDATE * param.data + (1.0 - cfg.TAU_SOFT_UPDATE) * target_param.data)

    
# --- Baseline Agents (Stubs) ---
class SPERCBaseAgent(SPERCAgent):
    # Overwrite the embedding function to use a simple flattened vector
    def _get_state_embedding(self, state):
        # Flatten everything into a single vector per node
        # This is a simplified example
        node_feats = torch.FloatTensor(state['node_features']).to(cfg.DEVICE)
        service_feats = torch.FloatTensor([state['service']['workload']]).repeat(cfg.N_NODES, 1).to(cfg.DEVICE)
        return torch.cat([node_feats, service_feats], dim=1)

class DDRLAgent(BaseAgent):
    def select_action(self, state, exploration=True):
        # Implement logic for the flattened D-DRL model
        return (random.randrange(cfg.N_NODES), np.random.uniform(0, 1))

class GreedyCPUAgent(BaseAgent):
    def select_action(self, state, **kwargs):
        available_cpus = state['node_features'][:, 0]
        best_node = np.argmax(available_cpus)
        # Allocate just enough CPU to meet latency, or max possible
        cpu_to_alloc = min(available_cpus[best_node], state['service']['workload'] / (state['service']['max_latency'] * 0.9))
        return (best_node, cpu_to_alloc)

class GreedyLSAgent(BaseAgent):
    def select_action(self, state, **kwargs):
        service_layers = state['service']['r_q']
        node_layers = state['layer_matrix']
        layer_sizes = self.p.layer_sizes # Assumes agent has access to this
        
        costs = []
        for i in range(cfg.N_NODES):
            missing_layers = service_layers * (1 - node_layers[i])
            cost = np.sum(missing_layers * layer_sizes)
            costs.append(cost)
        
        best_node = np.argmin(costs)
        # Allocate just enough CPU
        available_cpu = state['node_features'][best_node, 0]
        cpu_to_alloc = min(available_cpu, state['service']['workload'] / (state['service']['max_latency'] * 0.9))
        return (best_node, cpu_to_alloc)

class RandomAgent(BaseAgent):
    def select_action(self, state, **kwargs):
        node_idx = random.randrange(cfg.N_NODES)
        cpu_alloc = np.random.uniform(0, state['node_features'][node_idx, 0])
        return (node_idx, cpu_alloc)