import numpy as np
from sumo_rl import parallel_env
from algorithms.mappo import MAPPO
import warnings
import torch

warnings.filterwarnings("ignore", module="pettingzoo.utils.conversions")
# Force PyTorch to use Tensor Cores for matrix multiplications
torch.set_float32_matmul_precision('high')

# 1. Bring in YOUR custom reward function
def custom_reward_function(traffic_signal):
    total_queue = traffic_signal.get_total_queued()
    total_wait = sum(traffic_signal.get_accumulated_waiting_time_per_lane())
    # Penalize queue and wait time (scaled down)
    return -(total_queue + (total_wait / 100.0))

print("Initializing SUMO Environment with MAPPO Agent...")

# 2. Swap the dummy env for YOUR SUMO env
env = parallel_env(
    net_file='kottayam.net.xml',
    route_file='kottayam.rou.xml',
    use_gui=False, # Turn to False later for fast overnight training
    num_seconds=3600,
    reward_fn=custom_reward_function,
    out_csv_name='outputs/mappo_training_run'
)

# Training Hyperparameters
NUM_EPISODES = 20
ROLLOUT_LENGTH = 100 # Increased slightly for traffic flow

obs, infos = env.reset()

# Find maximum dimensions for BOTH observations and actions across all asymmetrical intersections
max_obs_dim = max([len(o) for o in obs.values()])
max_action_dim = max([env.action_space(a).n for a in env.agents])

agent = MAPPO(obs_dim=max_obs_dim, 
              action_dim=max_action_dim, 
              num_agents=len(env.agents))

episode_rewards = []

for episode in range(NUM_EPISODES):
    obs, infos = env.reset()
    total_reward = 0

    # 4. The Execution Loop
    for step in range(ROLLOUT_LENGTH):
        if not env.agents: # Stop if all traffic lights disappear
            break
            
        # Pad current observations
        padded_obs = {}
        for agent_id, o in obs.items():
            if len(o) < max_obs_dim:
                padded_obs[agent_id] = np.pad(o, (0, max_obs_dim - len(o)), 'constant')
            else:
                padded_obs[agent_id] = o

        # AI decides the traffic lights using the padded dimensions
        actions, log_probs = agent.get_actions(padded_obs)
        value, global_obs = agent.get_value(padded_obs)

        # BOUND THE ACTIONS: Prevent out-of-bounds errors for smaller intersections
        safe_actions = {}
        for agent_id, act in actions.items():
            # Uses modulo to ensure the action is always within the agent's specific allowed phases
            safe_actions[agent_id] = int(act) % env.action_space(agent_id).n

        # Environment steps forward using the validated safe_actions
        next_obs, rewards, terminations, truncations, infos = env.step(safe_actions)

        # Pad the next_obs before saving to the buffer
        padded_next_obs = {}
        if next_obs:
            for agent_id, n_o in next_obs.items():
                if len(n_o) < max_obs_dim:
                    padded_next_obs[agent_id] = np.pad(n_o, (0, max_obs_dim - len(n_o)), 'constant')
                else:
                    padded_next_obs[agent_id] = n_o
        else:
            padded_next_obs = next_obs

        if len(rewards) > 0:
            team_reward = np.mean(list(rewards.values()))
            total_reward += team_reward

            dones = {a: (terminations.get(a, False) or truncations.get(a, False)) for a in rewards.keys()}

            # Save memories to buffer (using the original un-bounded actions so log_probs remain accurate to the network)
            agent.store_transition(padded_obs, global_obs, actions, log_probs, rewards, dones, value)

        if not next_obs:
            break
            
        obs = next_obs

    print(f"\nEpisode {episode + 1} | Reward: {total_reward:.2f}")

    # Train the neural network
    result = agent.update()
    print(f"Actor Loss: {result['actor_loss']:.4f} | Critic Loss: {result['critic_loss']:.4f}")

    episode_rewards.append(total_reward)

# Save the trained brain!
torch.save(agent.actor.state_dict(), "model_kottayam.pth")
env.close()
print("\nTraining Complete! Check the outputs folder for CSV data.")