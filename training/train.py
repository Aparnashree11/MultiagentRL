import torch
import os
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
import numpy as np
from agents.multi_agent import MultiAgentRL
from training.replay_buffer import ReplayBuffer
from training.utils import save_checkpoint,load_checkpoint
from pettingzoo.atari import boxing_v2

preprocess = Compose([
    ToTensor(),               
    Resize((84, 84)),         
    Normalize(mean=[0.5], std=[0.5])  
])


def preprocess_observation(obs):
    """
    Preprocess a single observation (resize, normalize, and permute dimensions).
    """
    if isinstance(obs, torch.Tensor):
        if obs.max() > 1:  
            obs = obs / 255.0 
    else:
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32) / 255.0
        elif isinstance(obs, PIL.Image.Image):
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((84, 84)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            obs = preprocess(obs)  


    return obs.permute(2, 0, 1) if len(obs.shape) == 3 else obs  

def train(num_episodes=100, batch_size=32, target_update_freq=10, gamma=0.99, checkpoint_dir="checkpoints", num_eval_games=10):
    """
    Main training loop for multi-agent reinforcement learning with automatic evaluation post-training.
    """
  
    multi_agent = MultiAgentRL()
    multi_agent.env = boxing_v2.parallel_env(render_mode="human") 

    buffer = ReplayBuffer(capacity=10000) 
    agent1_checkpoint = os.path.join(checkpoint_dir, "agent1_latest.pth")
    agent2_checkpoint = os.path.join(checkpoint_dir, "agent2_latest.pth")

    if os.path.exists(agent1_checkpoint):
        print(f"Loading Agent 1 checkpoint from {agent1_checkpoint}")
        multi_agent.agent_1.q_network.load_state_dict(load_checkpoint(agent1_checkpoint))
        multi_agent.agent_1.update_target_network()

    if os.path.exists(agent2_checkpoint):
        print(f"Loading Agent 2 checkpoint from {agent2_checkpoint}")
        multi_agent.agent_2.q_network.load_state_dict(load_checkpoint(agent2_checkpoint))
        multi_agent.agent_2.update_target_network()

    for episode in range(num_episodes):
        obs_tuple = multi_agent.reset_environment() 
        obs_dict = obs_tuple[0]  


        obs_stack_1 = torch.cat([preprocess_observation(obs_dict["first_0"]).unsqueeze(0)] * 4, dim=0)
        obs_stack_2 = torch.cat([preprocess_observation(obs_dict["second_0"]).unsqueeze(0)] * 4, dim=0)

  
        done_flags = {"first_0": False, "second_0": False}
        episode_reward = {"first_0": 0, "second_0": 0}

        while not all(done_flags.values()):

            action_1 = multi_agent.agent_1.select_action(obs_stack_1, step=episode)
            action_2 = multi_agent.agent_2.select_action(obs_stack_2, step=episode)
            actions = {"first_0": action_1, "second_0": action_2}

            next_obs_tuple = multi_agent.step(actions)
            next_obs_dict, rewards_dict, terminations, truncations, infos = next_obs_tuple

            next_obs_stack_1 = torch.cat([obs_stack_1[1:], preprocess_observation(next_obs_dict["first_0"]).unsqueeze(0)], dim=0)
            next_obs_stack_2 = torch.cat([obs_stack_2[1:], preprocess_observation(next_obs_dict["second_0"]).unsqueeze(0)], dim=0)

            buffer.add(obs_stack_1.clone(), action_1, rewards_dict["first_0"], next_obs_stack_1.clone(), terminations["first_0"])
            buffer.add(obs_stack_2.clone(), action_2, rewards_dict["second_0"], next_obs_stack_2.clone(), terminations["second_0"])

            episode_reward["first_0"] += rewards_dict["first_0"]
            episode_reward["second_0"] += rewards_dict["second_0"]

            obs_stack_1 = next_obs_stack_1
            obs_stack_2 = next_obs_stack_2
            done_flags = terminations

            if len(buffer) > batch_size:
                batch = buffer.sample(batch_size)
                multi_agent.agent_1.update(batch)
                multi_agent.agent_2.update(batch)

        if episode % target_update_freq == 0:
            multi_agent.agent_1.update_target_network()
            multi_agent.agent_2.update_target_network()

        print(f"Episode {episode + 1}/{num_episodes} - Reward Agent 1: {episode_reward['first_0']}, Reward Agent 2: {episode_reward['second_0']}")

        if episode % 50 == 0:
            save_checkpoint(multi_agent.agent_1.q_network.state_dict(), os.path.join(checkpoint_dir, "agent1_latest.pth"))
            save_checkpoint(multi_agent.agent_2.q_network.state_dict(), os.path.join(checkpoint_dir, "agent2_latest.pth"))

    print("Training completed!")

    print(f"Starting automatic evaluation with {num_eval_games} games...")
    from training.evaluation import evaluate
    evaluate(num_eval_games=num_eval_games, checkpoint_dir=checkpoint_dir)



def evaluate(num_eval_games=1, checkpoint_dir="checkpoints"):
    """
    Evaluate the trained agents in the environment with GUI enabled.
    Loads the latest checkpoints for agents before evaluation.
    """
    multi_agent = MultiAgentRL()
    multi_agent.env = boxing_v2.parallel_env(render_mode="human")

    agent1_checkpoint = os.path.join(checkpoint_dir, "agent1_latest.pth")
    agent2_checkpoint = os.path.join(checkpoint_dir, "agent2_latest.pth")

    if os.path.exists(agent1_checkpoint):
        print(f"Loading Agent 1 checkpoint from {agent1_checkpoint}")
        multi_agent.agent_1.q_network.load_state_dict(load_checkpoint(agent1_checkpoint))
        multi_agent.agent_1.update_target_network() 

    if os.path.exists(agent2_checkpoint):
        print(f"Loading Agent 2 checkpoint from {agent2_checkpoint}")
        multi_agent.agent_2.q_network.load_state_dict(load_checkpoint(agent2_checkpoint))
        multi_agent.agent_2.update_target_network() 

    for game in range(num_eval_games):
        obs_tuple = multi_agent.reset_environment()
        obs_dict = obs_tuple[0]

        done_flags = {"first_0": False, "second_0": False}
        total_rewards = {"first_0": 0, "second_0": 0}

        while not all(done_flags.values()):
            action_1 = multi_agent.agent_1.select_action(obs_dict["first_0"], step=0, eval_mode=True)
            action_2 = multi_agent.agent_2.select_action(obs_dict["second_0"], step=0, eval_mode=True)

            actions = {"first_0": action_1, "second_0": action_2}
            obs_tuple = multi_agent.step(actions)

            obs_dict, rewards_dict, done_flags, truncations, infos = obs_tuple
            total_rewards["first_0"] += rewards_dict["first_0"]
            total_rewards["second_0"] += rewards_dict["second_0"]

        print(f"Game {game + 1} - Agent 1 Reward: {total_rewards['first_0']}, Agent 2 Reward: {total_rewards['second_0']}")

    print("Evaluation completed!")

