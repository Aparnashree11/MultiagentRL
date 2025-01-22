import torch
import os
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
import numpy as np
from agents.multi_agent import MultiAgentRL
from training.replay_buffer import ReplayBuffer
from training.utils import save_checkpoint,load_checkpoint
from pettingzoo.atari import boxing_v2
from training.train import preprocess_observation


def evaluate(num_eval_games=1, checkpoint_dir="checkpoints"):
    """
    Evaluate trained agents over multiple games and report average performance.

    Args:
        num_eval_games (int): Number of games to evaluate.
        checkpoint_dir (str): Directory containing saved model checkpoints.
    
    Returns:
        dict: Average rewards for both agents over evaluation games.
    """
    multi_agent = MultiAgentRL()
    multi_agent.env = boxing_v2.parallel_env(render_mode="human")  # Enable rendering for evaluation

    # Load agent checkpoints
    agent1_checkpoint = os.path.join(checkpoint_dir, "agent1_latest.pth")
    agent2_checkpoint = os.path.join(checkpoint_dir, "agent2_latest.pth")
    
    if os.path.exists(agent1_checkpoint):
        print(f"Loading Agent 1 checkpoint from {agent1_checkpoint}")
        multi_agent.agent_1.q_network.load_state_dict(load_checkpoint(agent1_checkpoint))
        multi_agent.agent_1.update_target_network()
    else:
        print("Agent 1 checkpoint not found. Using untrained Agent 1.")

    if os.path.exists(agent2_checkpoint):
        print(f"Loading Agent 2 checkpoint from {agent2_checkpoint}")
        multi_agent.agent_2.q_network.load_state_dict(load_checkpoint(agent2_checkpoint))
        multi_agent.agent_2.update_target_network()
    else:
        print("Agent 2 checkpoint not found. Using untrained Agent 2.")

    total_rewards = {"first_0": 0.0, "second_0": 0.0}

    for game in range(num_eval_games):
        print(f"Starting evaluation game {game + 1}/{num_eval_games}...")
        obs_tuple = multi_agent.reset_environment()
        obs_dict = obs_tuple[0]

        obs_stack_1 = torch.cat([preprocess_observation(obs_dict["first_0"]).unsqueeze(0)] * 4, dim=0)
        obs_stack_2 = torch.cat([preprocess_observation(obs_dict["second_0"]).unsqueeze(0)] * 4, dim=0)

        done_flags = {"first_0": False, "second_0": False}
        eval_step = 100 

        while not all(done_flags.values()):
            action_1 = multi_agent.agent_1.select_action(obs_stack_1, step=eval_step)
            action_2 = multi_agent.agent_2.select_action(obs_stack_2, step=eval_step)
            actions = {"first_0": action_1, "second_0": action_2}

            next_obs_tuple = multi_agent.step(actions)
            next_obs_dict, rewards_dict, terminations, truncations, infos = next_obs_tuple


            obs_stack_1 = torch.cat([obs_stack_1[1:], preprocess_observation(next_obs_dict["first_0"]).unsqueeze(0)], dim=0)
            obs_stack_2 = torch.cat([obs_stack_2[1:], preprocess_observation(next_obs_dict["second_0"]).unsqueeze(0)], dim=0)

    
            total_rewards["first_0"] += rewards_dict["first_0"]
            total_rewards["second_0"] += rewards_dict["second_0"]

    
            done_flags = {agent_id: terminations[agent_id] or truncations[agent_id] for agent_id in done_flags}

    avg_rewards = {key: total_rewards[key] / num_eval_games for key in total_rewards}
    print(f"Evaluation completed! Average Rewards: {avg_rewards}")
    return avg_rewards
