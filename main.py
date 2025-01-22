import argparse
from training.train import train
from training.evaluation import evaluate
import os

def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Reinforcement Learning with PettingZoo Atari Boxing")
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate'], default='train',
                        help="Mode to run the script: 'train' for training, 'evaluate' for evaluation.")
    
    parser.add_argument('--num_episodes', type=int, default=0,
                        help="Number of episodes to train or evaluate.")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size for experience replay during training.")
    parser.add_argument('--target_update_freq', type=int, default=10,
                        help="Frequency (in episodes) to update the target network.")
    parser.add_argument('--gamma', type=float, default=0.99,
                        help="Discount factor for Q-learning.")
    parser.add_argument('--num_eval_games', type=int, default=10,
                        help="Number of games to play during evaluation.")
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help="Directory to save model checkpoints during training.")
    
    args = parser.parse_args()
    if args.mode == 'train':
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        print(f"Checkpoint directory: {args.checkpoint_dir}")
    if args.mode == 'train':
        print(f"Starting training for {args.num_episodes} episodes...")
        train(num_episodes=args.num_episodes, 
      batch_size=args.batch_size, 
      target_update_freq=args.target_update_freq, 
      gamma=args.gamma,
      checkpoint_dir=args.checkpoint_dir)

    elif args.mode == 'evaluate':
        print(f"Evaluating agents over {args.num_eval_games} games...")
        evaluate(num_eval_games=args.num_eval_games, 
         checkpoint_dir=args.checkpoint_dir)

if __name__ == "__main__":
    main()
