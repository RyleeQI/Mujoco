import numpy as np
import torch
import gym
import argparse
import os
import d4rl
import wandb
import time
from tqdm import tqdm
import utils
import model


# Runs policy for X episodes and returns D4RL score
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, mean, std, seed_offset=100, eval_episodes=10):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + seed_offset)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			state = (np.array(state).reshape(1,-1) - mean)/std
			action = policy.select_action(state)
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes
	d4rl_score = eval_env.get_normalized_score(avg_reward) * 100

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}, D4RL score: {d4rl_score:.3f}")
	print("---------------------------------------")
	return d4rl_score

def eval_respolicy(policy, env_name, seed, mean, std, seed_offset=100, eval_episodes=10):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + seed_offset)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		# variant_shape_initial
		last_state = state
		last_action = policy.select_action(state)
		j = 0
		while not done:
			state = (np.array(state).reshape(1,-1) - mean)/std
			action = policy.select_action(state)
			if j != 0:
				resaction = policy.select_resaction(last_state, last_action, state, action)
				action = resaction + action
			last_state = state
			last_action = action
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward
			j += 1
			

	avg_reward /= eval_episodes
	d4rl_score = eval_env.get_normalized_score(avg_reward) * 100

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}, D4RL score: {d4rl_score:.3f}")
	print("---------------------------------------")
	return d4rl_score


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	# Experiment
	parser.add_argument("--policy", default="Res_TD3_BC_distil")               # Policy name
	parser.add_argument("--env", default="halfcheetah-expert-v2")        # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
	parser.add_argument("--phase0_timesteps", default=5e5, type=int)
	parser.add_argument("--distil_circle", default=1e4, type=int)
	parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
	# TD3
	parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99)                 # Discount factor
	parser.add_argument("--tau", default=0.005)                     # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	# TD3 + BC
	parser.add_argument("--alpha", default=2.5)
	parser.add_argument("--normalize", default=True)
	parser.add_argument("--en_wandb", type=int, default=0)                 # wandb
	parser.add_argument("--tag", type=str, default="test")

	args = parser.parse_args()
	print(args)
	
	tag_name = f"{args.policy}_{args.tag}_circle{args.distil_circle}_seed{args.seed}"
	if args.en_wandb == 1:
		wandb.login(key='e21fde8488217afdcac35175a2c2f20c0d19057a') #replace with your own key
		wandb.init(entity="metalearning", project="Res_Action_RL", group=args.env, name=tag_name)
		# save args & source code
		wandb.config.current_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
		wandb.config.update(args)
		wandb.save('./main.py')
		wandb.save('./model.py')
		wandb.save('./utils.py')
	
	str_env = '_'.join(f"{args.env}".split('-')[:])
	log_dir = os.path.join("./results/", str_env, tag_name)
	model_dir = os.path.join("./models/", str_env, tag_name)
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)

	file_name = f"{args.policy}_{str_env}_{args.seed}"
	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print(f"logdir: {log_dir}")
	print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	env = gym.make(args.env)

	# Set seeds
	env.seed(args.seed)
	env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
		# TD3
		"policy_noise": args.policy_noise * max_action,
		"noise_clip": args.noise_clip * max_action,
		"policy_freq": args.policy_freq,
		# TD3 + BC
		"alpha": args.alpha
	}

	# Initialize policy
	policy = model.Res_TD3_BC(**kwargs)

	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		policy.load(f"./models/{policy_file}")

	# replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
	# replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
	replay_buffer = utils.Enhanced_ReplayBuffer(args.env)
	if args.normalize:
		mean,std = replay_buffer.normalize_states() 
	else:
		mean,std = 0,1
	
	evaluations = []
	if args.en_wandb == 1:
		# pass
		wandb.watch(policy.resactor, log=all, log_freq=args.eval_freq)
		wandb.watch(policy.actor, log=all, log_freq=args.eval_freq)
		wandb.watch(policy.critic, log=all, log_freq=args.eval_freq)
	for t in tqdm(range(int(args.max_timesteps))):
		stage = -1
		factor = t % args.distil_circle
		if factor < args.phase0_timesteps/args.max_timesteps * args.distil_circle:
			actor_loss, critic_loss = policy.train_phase0(replay_buffer, args.batch_size)
			stage = 0
		else:
			actor_loss, critic_loss = policy.train_phase1(replay_buffer, args.batch_size)
			stage = 1
		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			print(f"Time steps: {t+1}")
			if stage == 0:
				evaluations.append(eval_policy(policy, args.env, args.seed, mean, std))
			elif stage == 1:
				evaluations.append(eval_respolicy(policy, args.env, args.seed, mean, std))
			np.save(f"./results/{file_name}", evaluations)
			if args.save_model: 
				policy.save(f"{model_dir}/{file_name}")
				if args.en_wandb == 1:
					wandb.save(f"{log_dir}/{file_name}"+ "_critic")
					wandb.save(f"{log_dir}/{file_name}"+ "_critic_optimizer")
					wandb.save(f"{log_dir}/{file_name}"+ "_actor")
					wandb.save(f"{log_dir}/{file_name}"+ "_actor_optimizer")
					wandb.save(f"{log_dir}/{file_name}"+ "_resactor")
					wandb.save(f"{log_dir}/{file_name}"+ "_resactor_optimizer")
		
			if args.en_wandb == 1:
					wandb.log({
						"Training Step": t+1,
						"Test Score": evaluations[-1],
						"Average Test Score": np.mean(evaluations[-10:]),
						"actor_loss": actor_loss, 
						"critic_loss": critic_loss, 
						})
