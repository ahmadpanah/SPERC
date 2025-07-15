# main.py

import numpy as np
import pandas as pd
from tqdm import tqdm

import config as cfg
from environment import EdgeEnv
from agents import SPERCAgent, SPERCBaseAgent, DDRLAgent, GreedyCPUAgent, GreedyLSAgent, RandomAgent

def train(agent, env):
    print(f"--- Training {agent.__class__.__name__} ---")
    for i_episode in tqdm(range(cfg.N_TRAINING_EPISODES)):
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state, exploration=True)
            next_state, reward, done, info = env.step(action)
            
            # RL agents store transitions and learn
            agent.store_transition(state, action, reward, next_state, done)
            agent.update()
            
            state = next_state

def evaluate(agent, env):
    total_revenue = 0
    total_latency = []
    total_storage_cost = []
    total_accepted = 0
    
    state = env.reset()
    done = False
    while not done:
        action = agent.select_action(state, exploration=False)
        next_state, reward, done, info = env.step(action)
        
        if info['accepted']:
            total_revenue += info['net_revenue']
            total_latency.append(info['latency'])
            total_storage_cost.append(info['storage_cost'])
        
        state = next_state
    
    avg_latency = np.mean(total_latency) if total_latency else 0
    avg_storage = np.mean(total_storage_cost) if total_storage_cost else 0
    acceptance_ratio = (env.accepted_services / cfg.N_SERVICES) * 100
    
    return total_revenue, avg_latency, avg_storage, acceptance_ratio

def main():
    # Instantiate Environment
    env = EdgeEnv(cfg)

    # Instantiate Agents
    agents = {
        "SPERC": SPERCAgent(cfg),
        "SPERC-Base": SPERCBaseAgent(cfg),
        "D-DRL": DDRLAgent(),
        "Greedy-CPU": GreedyCPUAgent(),
        "Greedy-LS": GreedyLSAgent(cfg), # Needs layer sizes
        "Random": RandomAgent()
    }

    # Train RL-based agents
    train(agents["SPERC"], env)
    train(agents["SPERC-Base"], env)
    # train(agents["D-DRL"], env) # Assuming D-DRL also needs training

    # --- Run Evaluation ---
    results = []
    print("\n--- Starting Evaluation ---")
    for name, agent in agents.items():
        print(f"Evaluating {name}...")
        run_revenues, run_latencies, run_storages, run_ratios = [], [], [], []
        
        for _ in tqdm(range(cfg.N_EVALUATION_RUNS)):
            # Create a new env instance for each run to ensure independence
            eval_env = EdgeEnv(cfg)
            # Pass env params to agents that need them
            if isinstance(agent, (GreedyLSAgent)):
                agent.p = eval_env.p
                
            rev, lat, stor, ratio = evaluate(agent, eval_env)
            run_revenues.append(rev)
            run_latencies.append(lat)
            run_storages.append(stor)
            run_ratios.append(ratio)
        
        results.append({
            "Algorithm": name,
            "Net Revenue": np.mean(run_revenues),
            "Avg. Latency (s)": np.mean(run_latencies),
            "Avg. Storage Cost (MB)": np.mean(run_storages),
            "Acceptance Ratio (%)": np.mean(run_ratios)
        })

    # Display results in a format similar to Table 4
    results_df = pd.DataFrame(results).set_index("Algorithm")
    print("\n--- Overall Performance Comparison ---")
    print(results_df.to_string(formatters={
        'Net Revenue': '{:,.1f}'.format,
        'Avg. Latency (s)': '{:,.2f}'.format,
        'Avg. Storage Cost (MB)': '{:,.1f}'.format,
        'Acceptance Ratio (%)': '{:,.1f}'.format
    }))


if __name__ == "__main__":
    main()