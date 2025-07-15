# environment.py

import numpy as np
import networkx as nx
import config

class EdgeEnv:
    def __init__(self, params):
        self.p = params
        self._setup_network()
        self._generate_services()
        self.reset()

    def _setup_network(self):
        # Generate network topology using Waxman model
        self.graph = nx.waxman_graph(self.p.N_NODES, beta=0.5, alpha=0.5)
        
        # Assign node capacities
        self.node_cpu_capacity = np.random.uniform(*self.p.CPU_CAPACITY_RANGE, size=self.p.N_NODES)
        self.node_storage_capacity = np.random.uniform(*self.p.STORAGE_CAPACITY_RANGE, size=self.p.N_NODES) * 1024 # Convert GB to MB

        # Assign bandwidth to links
        for (u, v) in self.graph.edges():
            self.graph.edges[u, v]['bandwidth'] = np.random.uniform(*self.p.BANDWIDTH_RANGE)

        # Pre-compute shortest paths for latency calculation
        self.shortest_paths = dict(nx.all_pairs_dijkstra_path(self.graph))
        
        # Initialize layer presence on nodes
        self.initial_layer_matrix = np.random.choice(
            [0, 1],
            size=(self.p.N_NODES, self.p.N_UNIQUE_LAYERS),
            p=[1 - self.p.INITIAL_LAYER_PRESENCE, self.p.INITIAL_LAYER_PRESENCE]
        )

    def _generate_services(self):
        self.services = []
        self.layer_sizes = np.random.uniform(*self.p.LAYER_SIZE_RANGE, size=self.p.N_UNIQUE_LAYERS)
        
        for _ in range(self.p.N_SERVICES):
            # Layer requirements for the service
            num_layers = np.random.randint(*self.p.LAYERS_PER_SERVICE_RANGE)
            required_layers_indices = np.random.choice(self.p.N_UNIQUE_LAYERS, num_layers, replace=False)
            r_q = np.zeros(self.p.N_UNIQUE_LAYERS)
            r_q[required_layers_indices] = 1
            
            # Workload and QoS
            workload = np.random.uniform(*self.p.SERVICE_WORKLOAD_RANGE) # Total workload in Giga-cycles
            max_latency = np.random.uniform(*self.p.MAX_LATENCY_RANGE)
            
            # Simplified workload distribution: assume it originates from a random node
            source_node = np.random.randint(0, self.p.N_NODES)

            self.services.append({
                'r_q': r_q,
                'workload': workload,
                'max_latency': max_latency,
                'revenue': self.p.BASE_REVENUE,
                'source_node': source_node
            })

    def reset(self):
        self.current_service_idx = 0
        self.available_cpu = np.copy(self.node_cpu_capacity)
        self.available_storage = np.copy(self.node_storage_capacity)
        self.layer_matrix = np.copy(self.initial_layer_matrix)
        self.accepted_services = 0
        return self._get_state()

    def _get_state(self):
        if self.current_service_idx >= self.p.N_SERVICES:
            return None # Episode finished
            
        service = self.services[self.current_service_idx]
        
        # Node features: [available_cpu, available_storage]
        node_features = np.vstack([self.available_cpu, self.available_storage]).T
        
        state = {
            'graph': self.graph,
            'node_features': node_features,
            'layer_matrix': self.layer_matrix,
            'service': service,
            'lambda': self.p.STRATEGIC_FACTOR_LAMBDA
        }
        return state

    def step(self, action):
        node_idx, cpu_allocation = action
        service = self.services[self.current_service_idx]

        # --- Check for validity and calculate costs ---
        is_valid = True
        
        # 1. CPU constraint
        if cpu_allocation <= 0 or cpu_allocation > self.available_cpu[node_idx]:
            is_valid = False

        # 2. Storage Cost & Constraint
        required_layers = service['r_q']
        node_layers = self.layer_matrix[node_idx]
        missing_layers_mask = required_layers * (1 - node_layers)
        storage_cost = np.sum(missing_layers_mask * self.layer_sizes) # in MB
        
        if storage_cost > self.available_storage[node_idx]:
            is_valid = False

        # 3. Latency Calculation & Constraint
        # Computation Latency
        comp_latency = service['workload'] / cpu_allocation if cpu_allocation > 0 else float('inf')
        
        # Transmission Latency (simplified model from paper)
        # Data size proportional to workload, sent from a source node
        data_size_mb = service['workload'] * 0.1 # Heuristic: 0.1 MB per Giga-cycle
        source_node = service['source_node']
        trans_latency = 0
        if source_node != node_idx:
            path = self.shortest_paths[source_node][node_idx]
            # Find bottleneck bandwidth on the path
            bottleneck_bw = min(self.graph.edges[u, v]['bandwidth'] for u, v in zip(path[:-1], path[1:]))
            trans_latency = (data_size_mb * 8) / bottleneck_bw # Time = Size_bits / BW_bps

        total_latency = comp_latency + trans_latency
        if total_latency > service['max_latency']:
            is_valid = False

        # --- Calculate Reward and Update State ---
        info = {
            'accepted': False, 
            'net_revenue': 0, 
            'latency': total_latency, 
            'storage_cost': storage_cost
        }
        
        if not is_valid:
            reward = -self.p.P_FAIL_REWARD
        else:
            # Update state for accepted service
            self.available_cpu[node_idx] -= cpu_allocation
            self.available_storage[node_idx] -= storage_cost
            self.layer_matrix[node_idx][missing_layers_mask == 1] = 1 # Update layers
            self.accepted_services += 1
            info['accepted'] = True

            # Calculate net revenue (Objective Function)
            latency_penalty = self.p.C1_LATENCY_COST_NORMALIZER * total_latency
            storage_penalty = self.p.C2_STORAGE_COST_NORMALIZER * storage_cost
            
            net_revenue = service['revenue'] - \
                          (self.p.STRATEGIC_FACTOR_LAMBDA * latency_penalty) - \
                          ((1 - self.p.STRATEGIC_FACTOR_LAMBDA) * storage_penalty)
            
            reward = net_revenue
            info['net_revenue'] = net_revenue

        self.current_service_idx += 1
        done = (self.current_service_idx >= self.p.N_SERVICES)
        next_state = self._get_state()
        
        return next_state, reward, done, info