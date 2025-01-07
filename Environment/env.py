import random
import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), './'))
from function_preprocessing import getFuntion, countHypervolume, domain
from benchmark_functions import set_noise_level, set_NERF_scene
import torch
from botorch import fit_gpytorch_mll
from botorch.models import SingleTaskGP, ModelListGP
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import RBFKernel, ScaleKernel, MaternKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import SumMarginalLogLikelihood
import math

NERF_CHAIRS = ["6838708", "6838711", "6838712", "6838713", "6838717", "6838718", "6838719", "6838721", "6838722", "6838723", "6838724", "6838729", "6838731", "6838732", "6838735", "6838738", "6838745", "6838746", "6838747", "6838750", "6838751", "6838752", "6838764", "6838766", "6838767", "6838770", "6838774", '6838701', '6838703', '6838704', '6838705', '6838706', '6838707', '6838710', '6838715', '6838716', '6838720', '6838726', '6838727', '6838728', '6838730', '6838733', '6838734', '6838736', '6838737', '6838739', '6838740', '6838742', '6838743', '6838748', '6838749', '6838753', '6838755', '6838756', '6838759', '6838760', '6838761', '6838762', '6838763', '6838768', '6838769', '6838771', '6838772', '6838773']

class GaussianProcess:
	def __init__(self, x, y, kernel, kernel_ls, f_num, dim = 1):
		self.GP = []
		self.kernel_ls = kernel_ls
		self.kernel = kernel
		self.f_num = f_num
		self.dim = dim
		train_X = torch.tensor(x, dtype=float)
		train_Y = torch.tensor(y, dtype=float)
		for i in range(f_num):
			if kernel[i] == "RBF":
				covar_module = RBFKernel(ard_num_dims = dim)
			elif kernel[i] == "matern52":
				covar_module = MaternKernel(ard_num_dims = dim)
			covar_module.lengthscale = kernel_ls[i]
			self.GP.append(SingleTaskGP(train_X=train_X, train_Y=train_Y[:,i].unsqueeze(1), covar_module = covar_module, outcome_transform=Standardize(m=1)))

	def fit(self, x, y):
		self.GP = []
		train_X = torch.tensor(x, dtype=float)
		train_Y = torch.tensor(y, dtype=float)
		ls = []
		for i in range(self.f_num):
			if self.kernel[i] == "RBF":
				covar_module = RBFKernel(ard_num_dims = self.dim)
			elif self.kernel[i] == "matern52":
				covar_module = MaternKernel(ard_num_dims = self.dim)
			covar_module.lengthscale = self.kernel_ls[i]
			self.GP.append(SingleTaskGP(train_X=train_X, train_Y=train_Y[:,i].unsqueeze(1), covar_module = covar_module, outcome_transform=Standardize(m=1)))
			mll = ExactMarginalLogLikelihood(likelihood=self.GP[i].likelihood, model=self.GP[i])
			fit_gpytorch_mll(mll)
			ls.append(self.GP[i].covar_module.lengthscale)
		return ls

	def construct_state_action_pair(self, domain, y_star, t):
		means = []
		variances = []
		for i in range(self.f_num):
			output = self.GP[i].posterior(torch.tensor(domain))
			means.append(output.mean.cpu().detach().numpy())
			variances.append(output.variance.cpu().detach().numpy())
		means = torch.tensor(np.array(means)).squeeze().T
		variances = torch.tensor(np.array(variances)).squeeze().T
		state_action_pairs = torch.cat((means, variances), dim = 1)
		state_action_pairs = torch.cat((state_action_pairs, torch.tile(torch.tensor([y_star]), (len(domain),1))), dim = 1)
		state_action_pairs = torch.cat((state_action_pairs, torch.tile(torch.tensor([t]), (len(domain),1))), dim = 1)
		return state_action_pairs.detach().numpy()

class Environment:
	def __init__(self, T, domain_size, f_num, function_type, yahpo_scenario=None, seed=0, 
			  new_reward = False, 
			  perturb_noise_level = 0.1,
			  observation_noise_level = 0.1,
			  ls_learned_freq = 10, 
			  online_ls = 0,
			  ls_weight = 1,
			  domain_dim = 1,
			  NERF_scene = "chair",
			  discrete = False):
		if NERF_scene == "chairs":
			set_NERF_scene(NERF_scene, NERF_CHAIRS[0])
		else:
			set_NERF_scene(NERF_scene)
		# store argument 
		self.T = T
		self.domain_size = domain_size
		self.f_num = f_num
		self.function_type = function_type
		self.seed = seed
		self.new_reward = new_reward
		self.perturb_noise_level = perturb_noise_level
		self.observation_noise_level = observation_noise_level
		self.episode = 0
		self.NERF_scene = NERF_scene
		set_noise_level(perturb_noise_level)
		self.ls_learned_freq = ls_learned_freq
		self.online_ls = online_ls
		self.ls_weight = ls_weight
		self.domain_dim = domain_dim
		self.discrete = discrete
		# reset history 
		self.history = dict()
		self.history["x"] = []
		self.history["y_observed"] = []
		self.history["y_true"] = []
		self.history["hypervolume_observed"] = [0]
		self.history["hypervolume_true"] = [0]
		self.history["ls_esti"] = []
		self.t = 0
		# set ransom seed
		if seed > 0:
			np.random.seed(seed)
			torch.manual_seed(seed)
			random.seed(seed)

		# update function
		self.X = domain(function_type, domain_size, seed, domain_dim, False)
		self.domain_dim = np.shape(self.X)[-1]
		self.ls = [torch.tensor([[0.1]*self.domain_dim]) for i in range(f_num)]
		
		self.F, self.pareto_front, self.min_function_values, self.kernel, self.kernel_ls = getFuntion(self.X, f_num = self.f_num, function_type =  self.function_type, dim = self.domain_dim)
		if online_ls:
			self.kernel = ["matern52"]*f_num
			self.kernel_ls = [torch.tensor([[0.1]*self.domain_dim]) for i in range(f_num)]
		self.history["ls_true"] = self.kernel_ls
		self.history["kernel_true"] = self.kernel

		self.history["pareto_front"] = self.pareto_front

	def reset(self, seed=0, episode=0):
		# clear history
		self.history = dict()
		self.history["x"] = []
		self.history["y_observed"] = []
		self.history["y_true"] = []
		self.history["hypervolume_observed"] = [0]
		self.history["hypervolume_true"] = [0]
		self.history["ls_true"] = []
		self.history["ls_esti"] = []
		self.history["actions"] = []
		self.episode = episode
		self.ls = [torch.tensor([[0.1]*self.domain_dim]) for i in range(self.f_num)]
		# set ransom seed
		if seed > 0:
			np.random.seed(seed)
			torch.manual_seed(seed)
			random.seed(seed)
		
		if self.NERF_scene == "chairs":
			set_NERF_scene(self.NERF_scene, NERF_CHAIRS[self.episode])

		# update function
		self.X = domain(self.function_type, self.domain_size, self.seed, self.domain_dim, self.discrete)

		self.F, self.pareto_front, self.min_function_values, self.kernel, self.kernel_ls = getFuntion(self.X, f_num = self.f_num, function_type =  self.function_type, dim = self.domain_dim)
		if self.online_ls:
			self.kernel = ["matern52"]*self.f_num
			self.kernel_ls = [torch.tensor([[0.1]*self.domain_dim]) for i in range(self.f_num)]
		self.history["ls_true"] = self.kernel_ls
		self.history["kernel_true"] = self.kernel

		self.history["pareto_front"] = self.pareto_front

	def getYt(self, x):
		self.history["x"].append(x)
		y_true = []
		y_observed = []
		for i in range(len(self.F)):
			y = float(self.F[i](x))
			if math.isnan(y):
				continue
			y_true.append(y)
			y_observed.append(y_true[-1] + np.random.normal(0, self.observation_noise_level, 1)[0])
		self.history["y_true"].append(y_true)
		self.history["y_observed"].append(y_observed)
		self.t = len(self.history["x"])

	def getReward(self):
		if self.f_num == 1:
			reward = max(self.history["y_observed"][:]) - max(self.history["y_observed"][:-1])
		else:
			self.history["hypervolume_observed"].append(countHypervolume(np.array(self.history["y_observed"]), np.array(self.min_function_values)))
			if self.new_reward == True:
				reward = (self.history["hypervolume_observed"][self.t] - self.history["hypervolume_observed"][self.t-1])/((1.1**self.f_num) - self.history["hypervolume_observed"][self.t])
			else:
				reward = self.history["hypervolume_observed"][self.t] - self.history["hypervolume_observed"][self.t-1]
		return reward

	def getRegret(self):
		if (self.f_num == 1):
			regret = self.domain_max_points - max(self.history["y_true"])
		else:
			self.history["hypervolume_true"].append(countHypervolume(np.array(self.history["y_true"]), np.array(self.min_function_values)))
			regret = self.history["pareto_front"] - self.history["hypervolume_true"][self.t]
		return regret

	def fit_gp(self, t, initial_ls = 0.1):
		train_X = torch.tensor(np.array(self.history["x"]), dtype=float)
		train_Y = torch.tensor(self.history["y_observed"], dtype=float)
		GP_list = [SingleTaskGP(train_X=train_X, train_Y=train_Y[:,i].unsqueeze(1), 
									outcome_transform=Standardize(1)) for i in range(self.f_num)]
		if self.online_ls == 0:
			for i in range(self.f_num):
				GP_list[i].covar_module.base_kernel.lengthscale = self.kernel_ls[i] # pre-compute ls
			return ModelListGP(*GP_list)
		else:
			self.ls = [torch.tensor([[initial_ls]*self.domain_dim]) for i in range(self.f_num)]
			for i in range(self.f_num):
				GP_list[i].covar_module.base_kernel.lengthscale = self.ls[i] # initial ls

		model = ModelListGP(*GP_list)
		# model.covar_module.base_kernel.lengthscale = torch.tensor([[0.001]*self.domain_dim*self.f_num]).resize(self.f_num,1,self.domain_dim) # important step that make the learned ls be correct
		# model.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-5))
		if t % self.ls_learned_freq == 0:
			mll = SumMarginalLogLikelihood(model.likelihood, model)# mll = ExactMarginalLogLikelihood(likelihood=model.likelihood, model=model)
			try:
				fit_gpytorch_mll(mll)
			except RuntimeError:
				print("Something wrong")
			for i in range(self.f_num):
				GP_list[i].covar_module.base_kernel.lengthscale *= self.ls_weight
				self.ls[i] = GP_list[i].covar_module.base_kernel.lengthscale
			self.history["ls_esti"].append(self.ls)
			output = model(*model.train_inputs)
			loss = -mll(output, model.train_targets)
			return model, loss.item()
		return model, None

	def step(self, x):
		self.getYt(x)
		reward = self.getReward()
		regret = self.getRegret()
		y_star = max(self.history["y_observed"])
		return y_star, float(reward), float(regret)

def construct_state_action_pair(domain, gp, y_star, t):
	output = gp.posterior(torch.tensor(domain))
	mean = output.mean.cpu()
	variance = output.variance.cpu()
	state_action_pairs = torch.cat((mean, variance), dim = 1)
	state_action_pairs = torch.cat((state_action_pairs, torch.tile(torch.tensor([y_star]), (len(domain),1))), dim = 1)
	state_action_pairs = torch.cat((state_action_pairs, torch.tile(torch.tensor([t]), (len(domain),1))), dim = 1)
	return state_action_pairs.detach().numpy()

if __name__ == '__main__':
	f_num = 2
	T = 10
	domain_size = 5
	a = Environment(T = 100, domain_size = 1000, f_num = 2, function_type = "NERF_synthetic", seed = 1)

	for i in range(500):
		y_star, reward, regret = a.step([random.random()])
		# gp = GaussianProcess(np.array(a.history["x"]), np.array(a.history["y_observed"]), a.kernel, a.kernel_ls, a.f_num)
		# state_action_pairs = gp.construct_state_action_pair(a.X, y_star, i/T)
		gp = a.fit_gp(i)
		state_action_pairs = construct_state_action_pair(a.X, gp, y_star, i/T)
		b = 0