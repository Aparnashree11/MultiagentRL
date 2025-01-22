# env_setup/env_wrapper.py

import cv2
import numpy as np
from pettingzoo.atari import boxing_v2
from gym.spaces import Box
import gym

class AtariEnvWrapper(gym.Wrapper):
    def __init__(self, env, frame_size=(84, 84), stack_frames=4):
        super(AtariEnvWrapper, self).__init__(env)
        self.frame_size = frame_size
        self.stack_frames = stack_frames

        # Define a new observation space with stacked grayscale frames
        self.observation_space = Box(
            low=0,
            high=255,
            shape=(stack_frames, frame_size[0], frame_size[1]),  # Shape: [stack_frames, height, width]
            dtype=np.uint8
        )
        
        # Initialize frame stack buffers for both agents
        self.frame_stack_1 = np.zeros((self.stack_frames, *self.frame_size), dtype=np.uint8)
        self.frame_stack_2 = np.zeros((self.stack_frames, *self.frame_size), dtype=np.uint8)

    def preprocess_frame(self, frame):
        """
        Preprocesses a single frame by converting it to grayscale and resizing it.
        
        Args:
            frame (np.array): The raw RGB frame from the environment.
        
        Returns:
            np.array: The preprocessed grayscale frame.
        """
        # Convert RGB frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Resize the grayscale frame to the target size (e.g., 84x84)
        resized_frame = cv2.resize(gray_frame, self.frame_size)
        
        return resized_frame

    def reset(self):
        """
        Resets the environment and returns initial stacked observations for both agents.
        
        Returns:
            dict: A dictionary containing stacked observations for both agents.
        """
        # Reset the environment and get initial observations for both agents
        obs = self.env.reset()

        # Preprocess initial observations for both agents
        processed_obs_first = self.preprocess_frame(obs['first_0'])
        processed_obs_second = self.preprocess_frame(obs['second_0'])

        # Stack the same initial observation across all frames for both agents
        for i in range(self.stack_frames):
            self.frame_stack_1[i] = processed_obs_first
            self.frame_stack_2[i] = processed_obs_second
        
        return {'first_0': self.frame_stack_1.copy(), 'second_0': self.frame_stack_2.copy()}

    def step(self, action):
        """
        Takes a step in the environment and returns updated stacked observations for both agents.
        
        Args:
            action (dict): A dictionary containing actions for both agents.
        
        Returns:
            tuple: A tuple containing updated stacked observations, rewards, done flags, and info.
        """
        # Take a step in the environment with both agents' actions
        obs, reward, done, info = self.env.step(action)

        # Preprocess new observations for both agents
        processed_obs_first = self.preprocess_frame(obs['first_0'])
        processed_obs_second = self.preprocess_frame(obs['second_0'])

        # Update the frame stack for both agents (shift frames and add new one at end)
        self.frame_stack_1[:-1] = self.frame_stack_1[1:]
        self.frame_stack_1[-1] = processed_obs_first

        self.frame_stack_2[:-1] = self.frame_stack_2[1:]
        self.frame_stack_2[-1] = processed_obs_second

        return {'first_0': self.frame_stack_1.copy(), 'second_0': self.frame_stack_2.copy()}, reward, done, info

def create_atari_env():
    """
    Creates and returns an instance of the wrapped Atari Boxing environment.
    
    Returns:
         AtariEnvWrapper: The wrapped Atari Boxing environment with preprocessing.
    """
    # Create PettingZoo Atari Boxing environment
    env = boxing_v2.parallel_env()
    
    # Wrap it with our custom wrapper for preprocessing
    wrapped_env = AtariEnvWrapper(env)
    
    return wrapped_env