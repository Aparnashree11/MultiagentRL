from pettingzoo.atari import boxing_v2
from agents.agent import DQNAgent
import numpy as np

class MultiAgentRL:
    def __init__(self):
        """
        Initialize two DQN agents for multi-agent reinforcement learning in PettingZoo's Boxing environment.
        
         - Agent 1 plays as Player 1 in Boxing.
         - Agent 2 plays as Player 2 in Boxing.
         """

        self.env = boxing_v2.parallel_env(render_mode="human")
         
        num_actions = self.env.action_space('first_0').n

      
        input_channels = 4 

        self.agent_1 = DQNAgent(input_channels=input_channels, num_actions=num_actions)
        self.agent_2 = DQNAgent(input_channels=input_channels, num_actions=num_actions)


        self.step_counter = 0

    def reset_environment(self):
        """
        Reset the environment and return initial observations for both agents.
        """
        return self.env.reset()

    def step(self, actions):
    
        obs_dict_next, rewards_dict, terminations, truncations, info = self.env.step(actions)
    

        if terminations is None:
            terminations = {agent: False for agent in self.env.agents}
    

        agent_positions = {
            "first_0": info.get("first_0_position"),
            "second_0": info.get("second_0_position")
        }
        punches = {
            "first_0": info.get("first_0_punches", 0),
            "second_0": info.get("second_0_punches", 0)
        }
        hits_received = {
            "first_0": info.get("first_0_hits_received", 0),
            "second_0": info.get("second_0_hits_received", 0)
        }
    
        
        winner = None
        if any(terminations.values()):
            if rewards_dict["first_0"] > rewards_dict["second_0"]:
                winner = "first_0"
            elif rewards_dict["second_0"] > rewards_dict["first_0"]:
                winner = "second_0"
    
        custom_rewards = calculate_rewards(
            obs_dict_next, rewards_dict, agent_positions, punches, hits_received, terminations, winner
        )
    
        return obs_dict_next, custom_rewards, terminations, truncations, info
    
    
    
    def train_agents(self, num_episodes=1000, batch_size=32, target_update_freq=100):
        """
        Main loop for training both agents in parallel mode. This can include logic such as replay buffer,
        updating networks periodically, etc., but is left simple here for clarity.
        """

        obs_dict = self.reset_environment()

        done_flags = {"first_0": False, "second_0": False}

    
        for episode in range(num_episodes):
            while not all(done_flags.values()):
                
                action_1 = self.agent_1.select_action(obs_dict["first_0"], step=self.step_counter) 
                action_2 = self.agent_2.select_action(obs_dict["second_0"], step=self.step_counter)

                actions = {"first_0": action_1, "second_0": action_2}

                obs_dict_next, rewards_dict, done_flags, info_dict = self.step(actions)

            
                self.agent_1.update((obs_dict["first_0"], action_1, rewards_dict["first_0"], obs_dict_next["first_0"], done_flags["first_0"]))
                self.agent_2.update((obs_dict["second_0"], action_2, rewards_dict["second_0"], obs_dict_next["second_0"], done_flags["second_0"]))

        
                obs_dict = obs_dict_next

            
                if self.step_counter % target_update_freq == 0:
                    self.agent_1.update_target_network()
                    self.agent_2.update_target_network()

                self.step_counter += 1

            if episode % 10 == 0:
                print(f"Episode {episode}/{num_episodes} completed.")

def calculate_rewards(obs_dict, rewards_dict, agent_positions, punches, hits_received, terminations, winner=None):
    custom_rewards = {"first_0": 0.0, "second_0": 0.0}
    
    custom_rewards["first_0"] += punches.get("first_0", 0) * 2000.0 
    custom_rewards["second_0"] += punches.get("second_0", 0) * 2000.0

    custom_rewards["first_0"] -= hits_received.get("first_0", 0) * 10.0  
    custom_rewards["second_0"] -= hits_received.get("second_0", 0) * 10.0

    if agent_positions["first_0"] is not None and agent_positions["second_0"] is not None:
    
        distance = np.linalg.norm(np.array(agent_positions["first_0"]) - np.array(agent_positions["second_0"]))
        
        proximity_reward = max(0, 1.0 / (distance + 1e-5)) * 100.0 
        custom_rewards["first_0"] += proximity_reward
        custom_rewards["second_0"] += proximity_reward

    if any(terminations.values()) and winner:
        if winner == "first_0":
            custom_rewards["first_0"] += 100000.0 
            custom_rewards["second_0"] -= 5.0 
        elif winner == "second_0":
            custom_rewards["second_0"] += 100000.0
            custom_rewards["first_0"] -= 5.0
            
    custom_rewards["first_0"] += rewards_dict.get("first_0", 0)
    custom_rewards["second_0"] += rewards_dict.get("second_0", 0)

    return custom_rewards
