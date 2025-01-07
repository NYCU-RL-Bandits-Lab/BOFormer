import tqdm
import argparse
import random
import numpy as np
import pandas as pd
import torch
import sys, os
from DQN_MAML import DQN, ReplayMemory, collect_trajectories
import Environment.env
import Environment.function_preprocessing
import sobol_seq
import pickle

if __name__ == '__main__':
	# torch.set_num_threads(8)
	# torch.set_num_interop_threads(8)
	if not os.path.exists(f"results/FSAF/"):
		os.mkdir(f"results/FSAF/")
	if not os.path.exists(f"actions/"):
		os.mkdir(f"actions")
	parser = argparse.ArgumentParser(description='Multi-objective BO')

	parser.add_argument('--device', type=int, default=0, help='gpu device')
	parser.add_argument('--alpha', type=float, default=0.8, help='0~1')
	parser.add_argument('--hidden', type=int, default=100, help='size of hidden layer of Q-network')
	parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
	parser.add_argument('--gamma', type=float, default=0.98, help='discount factor')
	parser.add_argument('--batch_size', type=int, default=128, help='batch size')
	parser.add_argument('--target_update', type=int, default=5, help='period of updating target network')
	parser.add_argument('--epsilon', type=float, default=0.1, help='epsilon greedy for DQN')
	parser.add_argument('--K', type=int, default=5, help='few shot step')
	parser.add_argument('--N', type=int, default=5, help='number of particles')
	parser.add_argument('--total_task', type=int, default=3, help='number of big T')
	parser.add_argument('--m_size', type=int, default=100, help='size of meta data')
	parser.add_argument('--sample_EHI_Gaussian_num', type=int, default=2, help='number of EHI monte-carlo')
	parser.add_argument('--sub_sampling', type=int, default=0, help='sub sample or not')
	parser.add_argument('--sub_sampling_num', type=int, default=100, help='number of sub sample')
	parser.add_argument('--use_demo', type=int, default=1, help='active demo policy or not')
	parser.add_argument('--early_terminate', type=int, default=0, help='flag of early terminate or not')
	parser.add_argument('--select_type', type=str, default="average", help='average or individual')
	parser.add_argument('--seed', type=int, default=0, help='random seed of numpy, torch and random')
	parser.add_argument('--function_type', type=str, default="NERF_synthetic", help='train, RBF_0.05, RBF_0.2, RBF_0.3, matern52_0.05, matern52_0.2, matern52_0.3, BC, AR, ARS, DRZ, Branin, Currin, YAHPO')
	parser.add_argument('--yahpo_scenario', type=str, default='lcbench', help='lcbench, rbv2_xgboost, rbv2_svm, rbv2_glmnet')
	parser.add_argument('--NERF_scene', type=str, default="lego", help='NERF_scene')
	parser.add_argument('--domain_size', type=int, default=1000, help='domain size')
	parser.add_argument('--domain_dim', type=int, default=2, help='domain dimension')
	parser.add_argument('--f_num', type=int, default=2, help='number of objective function')
	parser.add_argument('--T', type=int, default=100, help='total iteration')
	parser.add_argument('--episode', type=int, default=100, help='number of episodes')
	parser.add_argument('--load_model_episode', type=int, default=500, help='index of stored models episode')
	parser.add_argument('--ls_learned_freq', type=int, default=10, help='freq of learning ls')
	parser.add_argument('--perturb_noise_level', type=float, default=0.01, help='perturbed noise')
	parser.add_argument('--observation_noise_level', type=float, default=0.0, help='observation noise')
	args = parser.parse_args()
	MOBO_info = "domain_{}_fnum_{}_alpha_{}_hidden_{}_lr_{}_gamma_{}_batch_{}_target_{}_epsilon_{}_K_{}_N_{}_total_task_{}_m_size_{}_EHI_{}_use_demo_{}_early_terminate_{}_select_type_{}".format(
		args.domain_size, args.f_num, args.alpha, args.hidden, args.lr, args.gamma, args.batch_size, args.target_update, args.epsilon, 
		args.K, args.N, args.total_task, args.m_size, args.sample_EHI_Gaussian_num, args.use_demo, args.early_terminate, args.select_type)

	# Set GPU number
	torch.cuda.set_device(args.device)

	# Set random seed
	if args.seed >= 0:
		np.random.seed(args.seed)
		torch.manual_seed(args.seed)
		random.seed(args.seed)

	in_dim = args.f_num * 3 + 1
	learner = DQN(in_dim=in_dim, out_dim=1, hidden_size=args.hidden, seed=args.seed, gamma=args.gamma, device=args.device,
					lr=args.lr, epsilon=args.epsilon, batch_size=args.batch_size, target_update=args.target_update, MOBO_info=MOBO_info,
					total_task=args.total_task, N=args.N, K=args.K, use_demo=args.use_demo, early_terminate=args.early_terminate, select_type=args.select_type)

	env = Environment.env.Environment(T=args.T, 
								   domain_size=args.domain_size, 
								   f_num=args.f_num, 
								   function_type=args.function_type, 
								   yahpo_scenario=args.yahpo_scenario,
								   NERF_scene = args.NERF_scene,
								   new_reward = 0,
								   online_ls=1,
								   perturb_noise_level = args.perturb_noise_level,
                      			   observation_noise_level = args.observation_noise_level,
								   domain_dim = args.domain_dim,
								   seed=args.seed)

	# train
	if (args.function_type == "train"):
		learner.train = True
		learner.train_model(args.T, args.episode, env, args.sample_EHI_Gaussian_num, args.sub_sampling, args.sub_sampling_num, seed=1000)
		for n in range(learner.N):
			torch.save(learner.policy_net[n].state_dict(), "./FSAF_models/Episode_{}_particle_{}_{}.pth".format("final", n, MOBO_info))

	# test
	else:
		for n in range(learner.N):
			learner.policy_net[n].load_state_dict(torch.load("./FSAF_models/Episode_{}_particle_{}_{}.pth".format(args.load_model_episode, n, MOBO_info), map_location=torch.device(args.device)))
			learner.target_net[n].load_state_dict(torch.load("./FSAF_models/Episode_{}_particle_{}_{}.pth".format(args.load_model_episode, n, MOBO_info), map_location=torch.device(args.device)))
		# initialization
		env.reset(seed=args.seed, episode=0)

        # initial sample
		X = Environment.function_preprocessing.domain(args.function_type, args.domain_size, args.seed) 
		y_star, reward, regret = env.step(X[random.randint(0,args.domain_size-1)])
		gp, _ = env.fit_gp(0)
		state_actions = Environment.env.construct_state_action_pair(X, gp, y_star, 0)

		# Create meta data
		buffer = ReplayMemory(args.m_size)
		collect_trajectories(buffer, args.m_size, learner, env)

		# Few shot update
		theta = learner.get_theta()
		for _ in range(args.K):
			theta = learner.svgd(theta, buffer, None, use_demo=False) 
		learner.load_theta(theta)


		for e in range(args.episode):
			seed=args.seed+e*10
			env.reset(seed=seed, episode=e)
			env.history['info'] = str(args)
			learner.train = False
			hypervolumes = []
			regrets = []

			# give the initial point first
			X = Environment.function_preprocessing.domain(env.function_type, env.domain_size, env.seed) 
			y_star, reward, regret = env.step(X[random.randint(0,env.domain_size-1)])
			gp, _ = env.fit_gp(0)
			state_actions = Environment.env.construct_state_action_pair(X, gp, y_star, 0)
			N_m = 5
			N_local = int(args.domain_size / N_m)
			for t in tqdm.tqdm(range(1, args.T)):
				candidate = []
				actions = [learner.select_action(state_actions) for _ in range(N_m)]
				for action in actions:
					candidate.append(X[action] + 0.05*(sobol_seq.i4_sobol_generate(env.domain_dim, N_local, seed+t)-0.5))
				candidate = np.clip(np.concatenate(candidate, axis = 0), 0.0001, 1)
				state_action_pairs = Environment.env.construct_state_action_pair(candidate, gp, y_star, t/args.T)
				print("e: {}\tt: {}".format(e, t))
				action = learner.select_action(state_actions)
				# env update
				y_star, reward, regret = env.step(candidate[action])
				
				# learn ls for GP
				gp, _ = env.fit_gp(t)
					
				X = Environment.function_preprocessing.domain(args.function_type, args.domain_size, seed+t) 
				next_state_actions = Environment.env.construct_state_action_pair(X, gp, y_star, t/args.T)
				regrets.append(regret)

			if args.function_type == "YAHPO":
				filename = 'FSAF_load_{}_{}_{}_{}_per_{}_obs_{}_{}.pkl'.format(args.load_model_episode, MOBO_info, args.function_type, args.yahpo_scenario, args.perturb_noise_level, args.observation_noise_level, e)
			elif args.function_type.startswith("NERF"):
				filename = 'FSAF_load_{}_{}_{}_{}_per_{}_obs_{}_{}.pkl'.format(args.load_model_episode, MOBO_info, args.function_type, args.NERF_scene, args.perturb_noise_level, args.observation_noise_level, e)
			else:
				filename = 'FSAF_load_{}_{}_{}_{}.pkl'.format(args.load_model_episode, MOBO_info, args.function_type, e)
			with open(os.path.join(f'./results/FSAF', filename), 'wb') as f:
				pickle.dump(env.history, f)