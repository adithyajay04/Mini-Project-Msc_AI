import argparse
import torch
import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

# Helper function to redraw the graph in real-time
def update_live_plot(ax, fig, line, data):
    window = 50
    if len(data) > window:
        smoothed = np.convolve(data, np.ones(window)/window, mode='valid')
        line.set_xdata(range(len(smoothed)))
        line.set_ydata(smoothed)
    else:
        line.set_xdata(range(len(data)))
        line.set_ydata(data)
        
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", type=str, required=True, help="The map file")
    parser.add_argument("--port", type=int, required=True, help="TraCI port")
    parser.add_argument("--mode", type=str, default="ai", choices=["ai", "fixed"], help="AI or Fixed Timer")
    args = parser.parse_args()

    # Mappings updated to include the real-world Kottayam topology
    model_map = {
        "grid2x2.net.xml": "model_2x2.pth",
        "grid3x3.net.xml": "model_3x3.pth",
        "grid4x4.net.xml": "model_4x4.pth",
        "grid5x5.net.xml": "model_5x5.pth",
        "kottayam.net.xml": "model_kottayam.pth" 
    }
    
    route_map = {
        "grid2x2.net.xml": "routes_2x2.rou.xml",
        "grid3x3.net.xml": "routes_3x3.rou.xml",
        "grid4x4.net.xml": "routes_4x4.rou.xml",
        "grid5x5.net.xml": "routes_5x5.rou.xml",
        "kottayam.net.xml": "kottayam.rou.xml" 
    }

    route_file = route_map.get(args.net)
    if not route_file:
        print(f"Error: No route mapping found for {args.net}")
        return

    # ==========================================
    # 1. INITIALIZE ENVIRONMENT & VARIABLES
    # ==========================================
    if args.mode == "fixed":
        import traci
        import sumolib
        print(f"[{args.net}] Starting FIXED TIMER Baseline...")
        sumo_binary = sumolib.checkBinary('sumo-gui')
        sumo_cmd = [sumo_binary, "-n", args.net, "-r", route_file, "--start", "--quit-on-end"]
        traci.start(sumo_cmd)
        plot_color, plot_label = 'crimson', 'Standard Fixed Timer'
        
    else:
        from sumo_rl import parallel_env
        from algorithms.mappo import MAPPO
        brain_file = model_map.get(args.net)
        print(f"[{args.net}] Starting MAPPO AI environment...")
        env = parallel_env(net_file=args.net, route_file=route_file, use_gui=True, num_seconds=3600)
        obs, _ = env.reset()
        
        # Dynamic Dimension Padding Logic
        max_obs_dim = max([len(o) for o in obs.values()])
        max_action_dim = max([env.action_space(a).n for a in env.agents])
        
        agent = MAPPO(obs_dim=max_obs_dim, 
                      action_dim=max_action_dim, 
                      num_agents=len(env.agents))
        
        try:
            agent.actor.load_state_dict(torch.load(brain_file))
            agent.actor.eval()
        except FileNotFoundError:
            print(f"CRITICAL ERROR: Could not find {brain_file}. Did you train it?")
            env.close()
            return
        plot_color, plot_label = 'teal', 'MAPPO Agent'

    # ==========================================
    # 2. SETUP LIVE TELEMETRY GRAPH
    # ==========================================
    plt.ion() 
    fig, ax = plt.subplots(figsize=(8, 5))
    line, = ax.plot([], [], label=plot_label, color=plot_color, linewidth=2)
    
    ax.set_title(f'Live Traffic Performance: {args.net}\nMode: {plot_label}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Simulation Steps')
    ax.set_ylabel('System Reward (Higher = Less Wait Time)')
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')

    history_rewards = []
    step = 0

    # ==========================================
    # 3. LIVE EXECUTION LOOP
    # ==========================================
    try:
        if args.mode == "fixed":
            while step < 3600 and traci.simulation.getMinExpectedNumber() > 0:
                traci.simulationStep()
                wait_time = sum([traci.vehicle.getWaitingTime(veh) for veh in traci.vehicle.getIDList()])
                history_rewards.append(-wait_time)
                
                if step % 20 == 0:
                    update_live_plot(ax, fig, line, history_rewards)
                step += 1
        else:
            while env.agents:
                # Pad asymmetric observation spaces (critical for real-world maps like Kottayam)
                padded_obs = {}
                for agent_id, o in obs.items():
                    if len(o) < max_obs_dim:
                        padded_obs[agent_id] = np.pad(o, (0, max_obs_dim - len(o)), 'constant')
                    else:
                        padded_obs[agent_id] = o

                actions, _ = agent.get_actions(padded_obs)
                
                # BOUND THE ACTIONS: Prevent out-of-bounds errors for smaller intersections
                safe_actions = {}
                for agent_id, act in actions.items():
                    safe_actions[agent_id] = int(act) % env.action_space(agent_id).n

                # Step environment with bounded actions
                next_obs, rewards, _, _, _ = env.step(safe_actions)
                history_rewards.append(sum(rewards.values()))
                
                if step % 20 == 0:
                    update_live_plot(ax, fig, line, history_rewards)
                    
                step += 1
                if not next_obs:
                    break
                obs = next_obs
                
    except OSError:
        print(f"[{args.net}] Simulation interrupted by user.")

    # ==========================================
    # 4. CLEANUP AND SAVE
    # ==========================================
    try:
        if args.mode == "fixed":
            traci.close()
        else:
            env.close()
    except:
        pass 

    print(f"[{args.net}] Simulation complete. Saving final graph...")
    plt.ioff() 
    
    graph_filename = f"Graph_{args.mode}_{args.net.replace('.net.xml', '')}.png"
    plt.savefig(graph_filename, dpi=300)
    plt.show() 

if __name__ == "__main__":
    run()