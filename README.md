
# Atari Boxing Project

This repository implements a **multi-agent reinforcement learning system** for the Atari Boxing environment. The goal of this project is to create two intelligent agents that compete against each other in the boxing game, learn from their interactions, and improve their performance over time. The agents are trained using **deep Q-learning** and work in parallel to ensure dynamic and competitive gameplay.

---

## 📜 Table of Contents

- [Features](#features)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Technical Details](#technical-details)
- [Results](#results)
- [Future Work](#future-work)

---

## 🌟 Features

- **Multi-Agent System**: Two agents trained simultaneously in a competitive environment.
- **Deep Q-Learning**: Uses convolutional neural networks (CNNs) to process game frames and predict actions.
- **Replay Buffer**: Stores past experiences for efficient training and stability.
- **Environment Wrapper**: Simplifies interaction with the PettingZoo Atari Boxing environment.
- **Dynamic Exploration**: Implements epsilon-greedy exploration with adaptive decay.
- **Evaluation Pipeline**: Robust tools to evaluate agent performance and visualize results.
- **Checkpointing**: Save and load models to resume training or evaluation.

---

## 📁 Directory Structure

```
AtariBoxingProject/
├── agents/
│   ├── __init__.py         # Initializes the agents package
│   ├── agent.py            # DQNAgent class for single-agent functionality
│   ├── multi_agent.py      # Multi-agent logic for training two agents
├── checkpoints/            # Directory for saving and loading model checkpoints
├── env_setup/
│   ├── __init__.py         # Initializes the environment setup package
│   ├── env_wrapper.py      # Wrapper for PettingZoo Atari environment
│   ├── utils.py            # Helper functions for environment setup
├── models/
│   ├── __init__.py         # Initializes the models package
│   ├── cnn_model.py        # CNN architecture for feature extraction
├── roms/                   # Contains the game ROM (Atari Boxing files)
├── training/
│   ├── __init__.py         # Initializes the training package
│   ├── evaluation.py       # Evaluates trained agents' performance
│   ├── replay_buffer.py    # Experience replay buffer implementation
│   ├── train.py            # Training script for the multi-agent system
│   ├── utils.py            # Helper functions for the training process
├── main.py                 # Entry point for running the project
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
├── test.py                 # Unit tests for components
```

---

## 🛠️ Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-username/AtariBoxingProject.git
   cd AtariBoxingProject
   ```

2. **Install Dependencies**

   Use the `requirements.txt` file to install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. **Add ROMs**

   Place the necessary Atari Boxing ROM files in the `roms/` directory. Ensure you have legal access to these files.

4. **Operating System Limitations**

   - This project works only on **Mac/Linux/Unix** systems. **Windows systems** are not supported due to compatibility issues with the PettingZoo Atari environment.
   - The default code is configured for Mac systems. For Linux/Unix, ensure you replace the following lines:
     ```python
     env = boxing_v2.env(render_mode="human")
     ```
     with:
     ```python
     env = boxing_v2.env(render_mode="human", auto_rom_install_path="roms")
     ```
     Update this in the following files:
     - `multi_agent.py`
     - `cnn_model.py`
     - `train.py`
     - `evaluation.py`

---

## 🚀 Usage

### Training the Agents

To train the agents, run:

```bash
python main.py --mode train --num_episodes 1000 --batch_size 64 --target_update_freq 10 --gamma 0.99 --checkpoint_dir checkpoints
```

### Evaluating the Agents

To evaluate the agents after training, run:

```bash
python main.py --mode evaluate --num_eval_games 10 --checkpoint_dir checkpoints
```

### Running Tests

Run unit tests to ensure code functionality:

```bash
python test.py
```

### Arguments for `main.py`

| Argument             | Description                                      | Default          |
|----------------------|--------------------------------------------------|------------------|
| `--mode`             | Mode to run the script: `train` or `evaluate`   | `train`          |
| `--num_episodes`     | Number of episodes to train                     | `0`              |
| `--batch_size`       | Batch size for experience replay during training| `32`             |
| `--target_update_freq`| Frequency (in episodes) to update target network| `10`             |
| `--gamma`            | Discount factor for Q-learning                  | `0.99`           |
| `--num_eval_games`   | Number of games to play during evaluation       | `10`             |
| `--checkpoint_dir`   | Directory to save/load model checkpoints        | `checkpoints`    |

---

## ⚙️ Technical Details

### 1. **Multi-Agent Reinforcement Learning**

The project uses a multi-agent version of DQN where each agent is represented by a CNN and trains independently while competing in the shared environment.

### 2. **CNN Architecture**

The CNN processes game frames and extracts features. The architecture includes:
- 3 convolutional layers for feature extraction.
- Fully connected layers for decision-making.
- Rectified Linear Unit (ReLU) activation for non-linearity.

### 3. **Replay Buffer**

Stores past experiences (`state`, `action`, `reward`, `next_state`) and samples mini-batches for training to ensure better convergence.

### 4. **Environment Wrapper**

Simplifies interaction with the PettingZoo environment by:
- Normalizing observations.
- Stacking consecutive frames for temporal awareness.

---

## 📊 Results

- **Training Progress**: Agents progressively learn strategies to maximize their scores.
- **Competitive Dynamics**: Both agents adapt to each other's strategies, leading to engaging matches.
- **Model Performance**: The agents demonstrate high accuracy in predicting optimal actions after sufficient training.

**Sample Training Log**:

| Metric              | Value {White Agent, Black Agent} |
|---------------------|-----------------------------------|
| Episodes Trained    | 1000                             |
| Average WinRates    | {0.3521, 0.6749}                 |
| Final Epsilon       | 0.01                             |
| Average Reward      | {-1.7, 1.7}                      |

---

## 🛠️ Future Work

- Implement advanced algorithms like Proximal Policy Optimization (PPO).
- Extend to cooperative multi-agent environments.
- Add visualization tools for real-time performance tracking.

---
