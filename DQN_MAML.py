from cmath import nan
from copy import deepcopy
import tqdm
import math
import random
import numpy as np
from collections import namedtuple, deque
from itertools import count
from PIL import Image
import Environment.env
import Environment.function_preprocessing
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import torch.autograd as autograd
from torch.distributions import Categorical
from itertools import count

from ehi import ExpectedHypervolumeImprovement
torch.autograd.set_detect_anomaly(True)
Transition = namedtuple('Transition', ('state_action', 'action', 'next_state_actions', 'reward'))


class Network(torch.nn.Module):
	def __init__(self, state_in_dim: int, out_dim: int, hidden_size: int) -> None:
		super(Network, self).__init__()
		# define the model structure
		#torch.manual_seed(1)
		self.fc1_1 = torch.nn.Linear(state_in_dim, hidden_size)
		self.fc1_2 = torch.nn.Linear(hidden_size, hidden_size)
		self.fc1_3 = torch.nn.Linear(hidden_size, out_dim)

	def forward(self, x1: torch.cuda.FloatTensor) -> torch.cuda.FloatTensor:
		# forward performs the forward propagation
		x1 = torch.nn.functional.relu(self.fc1_1(x1))
		x1 = torch.nn.functional.relu(self.fc1_2(x1))
		x1 = self.fc1_3(x1)

		return x1


class ReplayMemory(object):
	def __init__(self, capacity: int) -> None:
		self.memory = deque([], maxlen=capacity)

	def push(self, *args: tuple) -> None:
		self.memory.append(Transition(*args))

	def sample(self, batch_size: int) -> list:
		if batch_size >= len(self.memory):
			return random.sample(self.memory, len(self.memory))
		else:
			return random.sample(self.memory, batch_size)
	def __len__(self) -> int:
		return len(self.memory)


def forward(x, theta, L, in_dim, out_dim, hidden_size):
	index = 0
	for i in range(2*L):
		if i % 2 == 0:	# weight
			if i == 0:			# first layer
				x = torch.matmul(x,torch.transpose(theta[index:index+in_dim*hidden_size].reshape(hidden_size, in_dim), 0, 1).clone())
				index += in_dim * hidden_size
			elif i == 2*L-2:	# final layer
				x = torch.matmul(x,torch.transpose(theta[index:index+hidden_size*out_dim].reshape(out_dim, hidden_size), 0, 1).clone())
				index += hidden_size * out_dim
			else:				# hidden layer
				x = torch.matmul(x,torch.transpose(theta[index:index+hidden_size*hidden_size].reshape(hidden_size, hidden_size), 0, 1).clone())
				index += hidden_size*hidden_size
		else: 			# bias
			if i < 2*L-1:		# final layer
				x = torch.nn.functional.relu(x + theta[index:index+hidden_size].clone())
				index += hidden_size
			else:				# hidden layer
				x = x + theta[index:].clone()
				index += 1
	if index != theta.size()[0]:
		raise ValueError

	return x


class DQN():
	def __init__(self, in_dim: int, out_dim: int, hidden_size: int, seed: int, alpha: float=0.1,
				 gamma: float=1.0, lr: float=0.01, epsilon=0.1, batch_size: int=128, target_update=5, device=0, MOBO_info="",
				 total_task=3, N=5, K=5, use_demo=True, early_terminate=False, select_type="average") -> None:
		# set random seed
		self.MOBO_info = MOBO_info
		self.device = device
		if seed >= 0:
			np.random.seed(seed)
			torch.manual_seed(seed)
			random.seed(seed)
		self.gamma = gamma
		self.lr = lr
		self.target_update = target_update
		self.epsilon = epsilon
		self.in_dim = in_dim
		self.out_dim = out_dim
		self.hidden_size = hidden_size
		self.f_num = int(in_dim/3)
		self.use_demo = use_demo
		self.early_terminate = early_terminate
		self.select_type = select_type

		''' meta learning '''
		self.total_task = total_task
		self.N = N
		self.K = K
		self.S = 1 
		self.L = 3		# number of layer
		self.alpha = alpha
		self.beta = 0.001
		self.reply_batch_size_Q = batch_size
		self.reply_batch_size_D = 1
		self.policy_net = []
		self.target_net = []
		for n in range(self.N):
			self.policy_net.append(Network(in_dim, out_dim, hidden_size).cuda(self.device))
			self.target_net.append(Network(in_dim, out_dim, hidden_size).cuda(self.device))
			self.target_net[n].load_state_dict(self.policy_net[n].state_dict())
			self.target_net[n].eval()
		self.memory_Q = [ReplayMemory(1000)] * (2 ** self.f_num)
		self.memory_D = [ReplayMemory(1000)] * (2 ** self.f_num)
		self.train = False
	
	def select_action(self, state_action_pairs: np.ndarray) -> float:
		state_action_pairs = torch.from_numpy(state_action_pairs).float().cuda(self.device)
		if random.random() > 1 - self.epsilon and self.train == True:
			return torch.tensor([[random.randrange(np.shape(state_action_pairs)[0])]], device=self.device, dtype=torch.long)

		with torch.no_grad():
			Qs = self.policy_net[0](state_action_pairs)
			if self.select_type == "average":
				for i in range(self.N):
					Qs = Qs + self.policy_net[i](state_action_pairs)

				dist = Categorical(logits=Qs.squeeze(1))
				action = dist.sample()
				return action.item()
			elif self.select_type == "individual": 
				max_index = []
				for i in range(self.N):
					max_index.append(self.policy_net[i](state_action_pairs).argmax().item())
				return random.choice(max_index)
			else:
				raise ValueError

	def load_theta(self, theta):
		for n in range(self.N):
			d = self.in_dim
			m = self.hidden_size
			self.policy_net[n].state_dict()['fc1_1.weight'][:] = theta[n][:d*m].reshape(m, d).cuda(self.device)
			self.policy_net[n].state_dict()['fc1_1.bias'][:] = theta[n][d*m:d*m+m].cuda(self.device)
			self.policy_net[n].state_dict()['fc1_2.weight'][:] = theta[n][d*m+m:d*m+m+m*m].reshape(m, m).cuda(self.device)
			self.policy_net[n].state_dict()['fc1_2.bias'][:] = theta[n][d*m+m+m*m:d*m+m+m*m+m].cuda(self.device)
			self.policy_net[n].state_dict()['fc1_3.weight'][:] = theta[n][d*m+m+m*m+m:d*m+m+m*m+m+m].cuda(self.device)
			self.policy_net[n].state_dict()['fc1_3.bias'][:] = theta[n][d*m+m+m*m+m+m:].cuda(self.device)

	def get_theta(self):
		theta_particles = []
		for n in range(self.N):
			theta_particles.append(torch.cat([p.flatten() for p in self.policy_net[n].parameters()]))

		return theta_particles

	def train_model(self, T, episode, env, sample_EHI_Gaussian_num, sub_sampling, sub_sampling_num, seed):
		for ep in range(episode):
			print("\ne:", ep)
			optimizers = []
			for n in range(self.N):
				optimizers.append(optim.Adam(self.policy_net[n].parameters(), lr=self.beta))
				optimizers[n].zero_grad()
			meta_loss = 0

			for task in range(self.total_task):	# Line 4
				env.reset(seed=seed+ep*10+task, new_ls=True)
				theta_particles = []
				for n in range(self.N):		# For each particle
					theta_particles.append(torch.cat([p.flatten() for p in self.policy_net[n].parameters()]).requires_grad_())

				# Decide which replay buffer be used here, and related to the order of the lengthscale
				temp = []
				for i in range(len(env.kernel_ls)):
					if env.kernel_ls[i] <= 0.2:
						temp.append(0)
					else:
						temp.append(1)
				buffer_index = 0
				for i in range(len(temp)):
					buffer_index += (2**i)*temp[i]

				for k in range(self.K):	# Line 6, inner loop
					# Define demo policy
					ehi = ExpectedHypervolumeImprovement(domain_num=env.domain_num, f_num=env.f_num, domain_min_points=env.domain_min_points, 
						sample_Gaussian_num=sample_EHI_Gaussian_num, sub_sampling=sub_sampling, sub_sampling_num=sub_sampling_num)
					env.reset(seed=seed+ep*10+task, new_ls=False)
					collect_trajectories(self.memory_Q[buffer_index], T, self, env, early_terminate=self.early_terminate)	# Line 7
					env.reset(seed=seed+ep*10+task, new_ls=False)
					collect_trajectories(self.memory_D[buffer_index], T, ehi, env, early_terminate=self.early_terminate)	# Line 8
					agent_buffer = self.memory_Q[buffer_index]
					demo_buffer = self.memory_D[buffer_index]
					# Line 10
					if k == 0:
						theta_tau_K = self.svgd(theta_particles, agent_buffer, demo_buffer, use_demo=self.use_demo)
					else:
						theta_tau_K = self.svgd(theta_tau_K, agent_buffer, demo_buffer, use_demo=self.use_demo)

				theta_tau_star = []
				for n in range(self.N):
					theta_tau_star.append(theta_tau_K[n].clone())

				for s in range(self.S):		# Line 13
					# Line 15
					if s == 0:
						theta_tau_S = self.svgd(theta_tau_star, agent_buffer, demo_buffer, use_demo=self.use_demo)
					else:
						theta_tau_S = self.svgd(theta_tau_S, agent_buffer, demo_buffer, use_demo=self.use_demo)

				for n in range(self.N):
					# Compute meta loss (9) in page 6
					meta_loss += torch.linalg.norm(theta_tau_K[n] - theta_tau_S[n].detach()) ** 2

			# Line 18, compute gradient
			# with autograd.detect_anomaly():
			meta_loss.backward()

			# Check the different of each particle after update 
			for n in range(self.N): # For each particle
				# Line 18, Adam optimizer
				optimizers[n].step()
			
			# store theta
			for n in range(self.N):
				torch.save(self.policy_net[n].state_dict(), "./model/MAML/step/Episode_{}_particle_{}_{}.pth".format(ep, n, self.MOBO_info))

			if ep % self.target_update == 0:
				for n in range(self.N):
					self.target_net[n].load_state_dict(self.policy_net[n].state_dict())

	def svgd(self, theta, agent_buffer, demo_buffer, use_demo=True):
		criterion = torch.nn.MSELoss()
		new_theta = []
		grad_list = []
		for n in range(self.N):	# for each particle
			# sample agent mini-batch
			transitions_Q = agent_buffer.sample(self.reply_batch_size_Q)
			batch = Transition(*zip(*transitions_Q))
			state_action_batch = torch.cat(batch.state_action).cuda(self.device)
			next_state_actions_batch = torch.cat(batch.next_state_actions).cuda(self.device)
			reward_batch = torch.cat(batch.reward).cuda(self.device)

			if use_demo:
				# sample demo mini-batch, check whether mix these two data
				transitions_D = demo_buffer.sample(self.reply_batch_size_D)
				batch = Transition(*zip(*transitions_D))
				d_state_action_batch = torch.cat(batch.state_action).cuda(self.device)

			grad = 0
			for i in range(self.N):
				state_action_value = forward(state_action_batch, theta[i], self.L, self.in_dim, self.out_dim, self.hidden_size)
				next_action_state_value = self.target_net[i](next_state_actions_batch).max(1)[0].detach()
				expected_state_action_value = (next_action_state_value * self.gamma) + reward_batch.unsqueeze(1)
				# (4)
				C = criterion(state_action_value, expected_state_action_value)

				if use_demo:
					# (5)
					log_pi = forward(d_state_action_batch, theta[i], self.L, self.in_dim, self.out_dim, self.hidden_size)
				else:
					log_pi = 0
				# (7), no need to set retain_graph=True
				rbf_value = self.rbf(theta[i], theta[n])
				grad += torch.autograd.grad(outputs=(-1 / self.alpha * C + log_pi) * rbf_value.detach() / self.N, inputs=theta[i], create_graph=True, retain_graph=True)[0].clamp_(-1, 1)
				if i != n:
					grad += torch.autograd.grad(outputs=rbf_value / self.N, inputs=theta[i], create_graph=True, retain_graph=True)[0].clamp_(-1, 1)
			grad_list.append(grad)
			
		for n in range(self.N):
			new_theta.append(self.gradient_ascent(theta[n], grad_list[n], self.lr))

		return new_theta

	def gradient_ascent(self, theta, grad, lr):
		theta += lr * grad
		return theta

	def rbf(self, x, y, ls=1):
		ans = torch.exp(-(torch.norm(x-y)**2) / (2*(ls**2)))
		# if ans == nan:
		# 	a = 1
		return ans


def collect_trajectories(buffer, T, p, env, early_terminate=False):
	# give the initial point first
	# initial sample
	X = Environment.function_preprocessing.domain(env.function_type, env.domain_size, env.seed) 
	y_star, reward, regret = env.step(X[random.randint(0,env.domain_size-1)])
	gp, _ = env.fit_gp(0)
	state_actions = Environment.env.construct_state_action_pair(X, gp, y_star, 0)
	for t in tqdm.tqdm(range(T)):

		action = p.select_action(state_actions)
		y_star, reward, regret = env.step(X[action])
		gp, _ = env.fit_gp(t)
		X = Environment.function_preprocessing.domain(env.function_type, env.domain_size, env.seed+t) 
		next_state_actions = Environment.env.construct_state_action_pair(X, gp, y_star, t/T)
		
		if t != 0:
			buffer.push(torch.from_numpy(np.array([state_actions[action]])).float(),
						torch.tensor([action]).float(),
						torch.from_numpy(np.array([next_state_actions])).float(),
						torch.tensor([-regret]).float())
		if early_terminate and regret == 0:
			return
		state_actions = next_state_actions