import numpy as np
import torch
from tqdm import tqdm
import h5py
import os


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)


	def convert_D4RL(self, dataset):
		self.state = dataset['observations']
		self.action = dataset['actions']
		self.next_state = dataset['next_observations']
		self.reward = dataset['rewards'].reshape(-1,1)
		self.not_done = 1. - dataset['terminals'].reshape(-1,1)
		self.size = self.state.shape[0]


	def normalize_states(self, eps = 1e-3):
		mean = self.state.mean(0,keepdims=True)
		std = self.state.std(0,keepdims=True) + eps
		self.state = (self.state - mean)/std
		self.next_state = (self.next_state - mean)/std
		return mean, std



class Generate_ReplayBuffer(object):
	def __init__(self):
		self.size = 0
		tmp = []
		self.last_state = np.array(tmp)
		self.last_action = np.array(tmp)
		self.last_reward = np.array(tmp)

		self.state = np.array(tmp)
		self.action = np.array(tmp)
		self.reward = np.array(tmp)

		self.next_state = np.array(tmp)
		self.not_done = np.array(tmp)

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	def myappend(self, i, last_numpy, to_be_append):
		if i == 0:
			last_numpy = np.append(last_numpy, to_be_append)[np.newaxis,:]
		else:
			last_numpy = np.concatenate((last_numpy, to_be_append[np.newaxis,:]), axis=0)
		
		return last_numpy

	def convert_D4RL(self, dataset):
		print(dataset['observations'].shape)
		rewards_pre = dataset['rewards'].reshape(-1,1)
		not_done_pre = 1. - dataset['terminals'].reshape(-1,1)

		# for i in tqdm(range(dataset['observations'].shape[0] - 1 )):
		for i in tqdm(range(2000)):
			if np.linalg.norm(dataset['next_observations'][i] - dataset['observations'][i + 1]) > 0:
				pass
			    #TODO
			else:
				self.last_state = self.myappend(i = i, last_numpy = self.last_state, to_be_append = dataset['observations'][i])
				self.last_action = self.myappend(i = i, last_numpy = self.last_action, to_be_append = dataset['actions'][i])
				self.last_reward = self.myappend(i = i, last_numpy = self.last_reward, to_be_append = rewards_pre[i])

				self.state = self.myappend(i = i, last_numpy = self.state, to_be_append = dataset['observations'][i + 1])
				self.action = self.myappend(i = i, last_numpy = self.action, to_be_append = dataset['actions'][i + 1])
				self.reward = self.myappend(i = i, last_numpy = self.reward, to_be_append = rewards_pre[i + 1])
				
				self.next_state = self.myappend(i = i, last_numpy = self.next_state, to_be_append = dataset['next_observations'][i + 1])
				self.not_done = self.myappend(i = i, last_numpy = self.not_done, to_be_append = not_done_pre[i + 1])

		self.size = self.state.shape[0]

	def write_dataset(self, env):
		if not os.path.exists("./mydataset/"):
			os.makedirs("./mydataset/")

		h5f = h5py.File("./mydataset/" + env + '.h5', 'w')

		h5f.create_dataset('last_state', data=self.last_state)
		h5f.create_dataset('last_action', data=self.last_action)
		h5f.create_dataset('last_reward', data=self.last_reward)

		h5f.create_dataset('state', data=self.state)
		h5f.create_dataset('action', data=self.action)
		h5f.create_dataset('reward', data=self.reward)

		h5f.create_dataset('next_state', data=self.next_state)
		h5f.create_dataset('not_done', data=self.not_done)



class Enhanced_ReplayBuffer(object):
	def __init__(self, env):
		# with h5py.File('name-of-file.h5', 'r') as h5f:
		h5f = h5py.File("../mydataset/" + env + '.h5', 'r')
		self.last_state = np.array(h5f['last_state'])
		self.last_action = np.array(h5f['last_action'])
		self.last_reward = np.array(h5f['last_reward'])

		self.state = np.array(h5f['state'])
		self.action = np.array(h5f['action'])
		self.reward = np.array(h5f['reward'])

		self.next_state = np.array(h5f['next_state'])
		self.not_done = np.array(h5f['not_done'])
		self.size = self.state.shape[0]
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		print("load successfully!", "load " + str(self.state.shape[0]) +" trajectories!")
		print(type(self.state.shape))
		print(self.last_state.shape)
		print(self.last_action.shape)
		print(self.last_reward.shape)

		print(self.state.shape)
		print(self.action.shape)
		print(self.reward.shape)

		print(self.next_state.shape)
		print(self.not_done.shape)
		print(self.size)
  
	def normalize_states(self, eps = 1e-3):
		mean = self.state.mean(0,keepdims=True)
		std = self.state.std(0,keepdims=True) + eps
		self.last_state = (self.last_state - mean)/std
		self.state = (self.state - mean)/std
		self.next_state = (self.next_state - mean)/std
		return mean, std

	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			#t - 1
			torch.FloatTensor(self.last_state[ind]).to(self.device),
			torch.FloatTensor(self.last_action[ind]).to(self.device),
			torch.FloatTensor(self.last_reward[ind]).to(self.device),
			#t
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
   			torch.FloatTensor(self.reward[ind]).to(self.device),
			#t + 1
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)

