import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.cnn_model import CNNModel

class DQNAgent:
    def __init__(self, input_channels, num_actions, lr=1e-4, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=500):
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        self.q_network = CNNModel(input_channels=12, num_actions=num_actions)
        self.target_network = CNNModel(input_channels=12, num_actions=num_actions)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network.to(self.device)
        self.target_network.to(self.device)
        
    def select_action(self, state, step):
        epsilon_threshold = self.epsilon_end + (self.epsilon - self.epsilon_end) * np.exp(-1. * step / self.epsilon_decay)
        if np.random.rand() < epsilon_threshold:
            return np.random.randint(0, self.num_actions)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state)
            return q_values.argmax().item()

    def update(self, batch):
        states, actions, rewards, next_states, dones = batch
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_network(next_states).max(1)[0]
        targets = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = self.loss_fn(q_values, targets.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
