import torch
import warnings
from sumo_rl import parallel_env
from algorithms.mappo import MAPPO

warnings.filterwarnings("ignore", module="pettingzoo.utils.conversions")

print("Initializing Live AI Demonstration...")

# 1. Boot up SUMO with the GUI Turned ON
env = parallel_env(
    net_file='my_grid.net.xml',
    route_file='my_routes.rou.xml',
    use_gui=True,
    num_seconds=3600,
)

obs, infos = env.reset()
first_agent = env.agents[0]

# 2. Rebuild the neural network architecture
agent = MAPPO(
    obs_dim=len(obs[first_agent]),
    action_dim=env.action_space(first_agent).n,
    num_agents=len(env.agents)
)

# 3. Load your trained PyTorch weights into the brain
# (Make sure the filename matches what you saved!)
try:
    agent.actor.load_state_dict(torch.load("trained_actor.pth"))
    agent.actor.eval() # Set the network to "Evaluation Mode" (No learning)
    print("Trained brain loaded successfully!")
except FileNotFoundError:
    print("ERROR: Could not find 'trained_actor.pth'. Make sure you saved the model!")
    exit()

print("\nStarting simulation! Switch to the SUMO GUI window.")

# 4. The Live Execution Loop
while env.agents:
    # Ask the AI for its decisions based on the current traffic
    # We don't need log_probs or values here, just the pure actions
    actions, _ = agent.get_actions(obs)
    
    # Step the environment forward
    next_obs, rewards, terminations, truncations, infos = env.step(actions)
    
    if not next_obs:
        break
        
    obs = next_obs

env.close()
print("Demonstration Complete!")