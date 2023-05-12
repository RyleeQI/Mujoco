import argparse
import d4rl
import gym
import utils
import numpy as np

if __name__ == "__main__":
	
    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--env", default="halfcheetah-expert-v2")        # OpenAI gym environment name
    args = parser.parse_args()
    print(args)

    env = gym.make(args.env)

    replay_buffer = utils.Generate_ReplayBuffer()
    replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
    replay_buffer.write_dataset(env= args.env)

    another_buffer = utils.Enhanced_ReplayBuffer(args.env)