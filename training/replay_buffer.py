# training/replay_buffer.py

import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        """
        Initialize a replay buffer with a fixed capacity.

        Args:
            capacity (int): Maximum number of transitions to store in the buffer.
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        """
        Add a transition to the replay buffer.

        Args:
            state: The current state.
            action: The action taken.
            reward: The reward received.
            next_state: The next state after taking the action.
            done: Whether the episode has ended.
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Sample a random batch of transitions from the buffer.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            tuple: Batch of states, actions, rewards, next states and dones.
        """
        batch = random.sample(self.buffer, batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return np.stack(states), actions, rewards, np.stack(next_states), dones

    def __len__(self):
        return len(self.buffer)