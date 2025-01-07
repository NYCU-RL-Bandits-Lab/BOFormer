import numpy as np
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.optim.optimize import optimize_acqf
import collections
import pickle
from tqdm import tqdm
import torch
import os
import sys
import argparse
import random
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, parent_dir)
import Environment.env
import Environment.function_preprocessing

def gen_data(args):
    episodes = args.episode
    f_num = args.f_num
    domain_num = args.domain_size
    T = args.T
    function_type = args.function_type
    data = []
    env = Environment.env.Environment(T=T, domain_size=domain_num, f_num=f_num, function_type=function_type, seed=0, online_ls=0, new_reward=True, noise_level=0.1)
    for e in range(episodes):
        seed=e*10+1000
        episode_data = {}
        for k in ['observations', 'next_observations', 'actions', 'rewards', 'terminals']:
            episode_data[k] = []
        env.reset(seed=seed)
        X = Environment.function_preprocessing.domain(function_type, domain_num, seed) 
        action = random.randint(0,domain_num-1)
        y_star, reward, regret = env.step(X[action])
        gp = Environment.env.GaussianProcess(np.array(env.history["x"]), np.array(env.history["y_observed"]), env.kernel, env.kernel_ls, env.f_num)
        state_actions = gp.construct_state_action_pair(X, y_star, 0/args.T)

        episode_data['observations'].append(np.zeros(shape=state_actions.shape).reshape(-1).astype('float32'))
        episode_data['next_observations'].append(state_actions.reshape(-1).astype('float32'))
        episode_data['actions'].append(np.array([action]))
        episode_data['rewards'].append(reward)
        episode_data['terminals'].append(False)

        gp = env.fit_gp(0)
        pred = gp.posterior(torch.tensor(X, dtype=torch.double)).mean # gp(torch.tensor(X)).mean.T
        partitioning = FastNondominatedPartitioning(
            ref_point=torch.tensor(env.min_function_values, dtype=torch.double),
            Y=pred)

        standard_bounds = torch.zeros(2, env.domain_dim)
        standard_bounds[1] = 1
        
        
        print("e: {}".format(e))
        for t in tqdm(range(1, T)):
            if e % 2:
                action = random.randint(0,domain_num-1)
            else:
                gp = env.fit_gp(t)
                pred = gp.posterior(torch.tensor(X, dtype=torch.double)).mean # gp(torch.tensor(X)).mean.T
                partitioning = FastNondominatedPartitioning(
                    ref_point=torch.tensor(env.min_function_values, dtype=torch.double),
                    Y=pred)
                learner = qExpectedHypervolumeImprovement(
					model=gp,
					ref_point=torch.tensor(env.min_function_values, dtype=torch.double),
					partitioning=partitioning,
				)
                candidates, _ = optimize_acqf(
                    acq_function=learner,
                    bounds=standard_bounds.double(),
                    q=1,
                    num_restarts=1,
                    raw_samples=1,
                    options={"batch_limit": 1, "maxiter": 10},
                    sequential=True,
                )
                action = candidates.detach().numpy()[0]
                action = (X-action).argmin()

            y_star, reward, regret = env.step(X[action])
            X = Environment.function_preprocessing.domain(args.function_type, args.domain_size, seed+t)
            gp = Environment.env.GaussianProcess(np.array(env.history["x"]), np.array(env.history["y_observed"]), env.kernel, env.kernel_ls, env.f_num)
            next_state_actions = gp.construct_state_action_pair(X, y_star, t/args.T) 
            
            episode_data['observations'].append(state_actions.reshape(-1).astype('float32'))
            episode_data['next_observations'].append(next_state_actions.reshape(-1).astype('float32'))
            episode_data['actions'].append(np.array([action]))
            episode_data['rewards'].append(reward)
            episode_data['terminals'].append(False)
            state_actions = next_state_actions
        for k in ['observations', 'next_observations', 'actions', 'rewards', 'terminals']:
            episode_data[k] = np.stack(episode_data[k])
            
        data.append(episode_data)
        
    path = 'Decision_Transformer/decision_transformer/envs/data'
    os.makedirs(path, exist_ok=True)
    with open(path+'/{}_f{}_domain{}_ep{}.pkl'.format(function_type, f_num, domain_num, episodes), 'wb') as f:
        pickle.dump(data, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-objective BO')
    parser.add_argument('--episode', type=int, default=100)
    parser.add_argument('--T', type=int, default=100)
    parser.add_argument('--domain_size', type=int, default=1000)
    parser.add_argument('--f_num', type=int, default=2)
    parser.add_argument('--function_type', type=str, default='train_large')
    args = parser.parse_args()
    gen_data(args)