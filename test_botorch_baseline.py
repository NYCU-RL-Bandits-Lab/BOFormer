import os
import torch
import random
import numpy as np
import argparse
from tqdm import tqdm
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.optim.optimize import optimize_acqf
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement, qNoisyExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.joint_entropy_search import qLowerBoundMultiObjectiveJointEntropySearch
from botorch.acquisition.multi_objective.hypervolume_knowledge_gradient import qHypervolumeKnowledgeGradient
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.utils.sampling import sample_simplex
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.acquisition.objective import GenericMCObjective
from botorch.models.model_list_gp_regression import ModelListGP
import sys, os
import Environment.env
import Environment.function_preprocessing
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.gp_regression import SingleTaskGP
from gpytorch.kernels import RBFKernel, ScaleKernel
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.models.transforms.outcome import Standardize
import warnings
from gpytorch.constraints import GreaterThan
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_mll
sys.path.append(os.path.join(os.path.dirname(__file__), './'))
import JES_utils
import pickle
from gpytorch.mlls import SumMarginalLogLikelihood

warnings.filterwarnings("ignore")

if __name__=='__main__':
    os.makedirs(f"results/", exist_ok=True)
    parser = argparse.ArgumentParser(description='Multi-objective BO')
    parser.add_argument('--function_type', type=str, default="AR", help='RBF_0.05, RBF_0.2, RBF_0.3, matern52_0.05, matern52_0.2, matern52_0.3, BC, AR, ARS, DRZ, Branin, Currin, YAHPO')
    parser.add_argument('--yahpo_scenario', type=str, default='lcbench', help='lcbench, rbv2_xgboost, rbv2_svm, rbv2_glmnet')
    parser.add_argument('--domain_size', type=int, default=1000, help='domain size')
    parser.add_argument('--f_num', type=int, default=2, help='number of objective function')
    parser.add_argument('--T', type=int, default=30, help='total iteration')
    parser.add_argument('--episode', type=int, default=100, help='number of episodes')
    parser.add_argument('--ls_learned_freq', type=int, default=10, help='freq of learning ls')
    parser.add_argument('--learner', type=str, default='qNEHVI', help='qEHVI, qNEHVI, qHVKG, qParEGO, JES')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--perturb_noise_level', type=float, default=0.1, help='perturbed noise')
    parser.add_argument('--observation_noise_level', type=float, default=0.1, help='observation noise')
    parser.add_argument('--online_ls', type=int, default=1, help='ls in testing')
    parser.add_argument('--domain_dim', type=int, default=2, help='domain dimension')
    parser.add_argument('--NERF_scene', type=str, default="ship", help='NERF_scene')
    parser.add_argument('--discrete', type=int, default=0, help='discrete')
    args = parser.parse_args()

    if "RBF" in args.function_type or "matern" in args.function_type:
        args.domain_dim = 1

    env = Environment.env.Environment(T = args.T, 
                      domain_size = args.domain_size, 
                      f_num = args.f_num, 
                      function_type = args.function_type,
                      yahpo_scenario= args.yahpo_scenario, 
                      seed = args.seed,
                      new_reward = 0,
                      perturb_noise_level = args.perturb_noise_level,
                      observation_noise_level = args.observation_noise_level,
                      ls_learned_freq = args.ls_learned_freq,
                      online_ls = args.online_ls,
                      domain_dim = args.domain_dim,
                      NERF_scene = args.NERF_scene,
                      discrete = args.discrete)
    
    for e in range(args.episode):
        # initialization
        seed=args.seed+e*10
        env.reset(seed=seed, episode=e)
        env.history['info'] = str(args)
        regrets = []
        losses = []
        actions_record = []

        # initial sample
        X = Environment.function_preprocessing.domain(args.function_type, args.domain_size, seed, d = args.domain_dim, discrete=args.discrete)
        if args.function_type == "NERF_synthetic":
            y_star, reward, regret = env.step(X[random.randint(0,np.shape(X)[0]-1)])
        else:
            y_star, reward, regret = env.step(X[random.randint(0,min(np.shape(X)[0]-1,args.domain_size-1))])

        gp, _ = env.fit_gp(0)
        pred = gp.posterior(torch.tensor(X, dtype=torch.double)).mean # gp(torch.tensor(X)).mean.T
        partitioning = FastNondominatedPartitioning(
            ref_point=torch.tensor(env.min_function_values, dtype=torch.double),
            Y=pred)
        
        standard_bounds = torch.zeros(2, env.domain_dim)
        standard_bounds[1] = 1

        for t in tqdm(range(1, args.T)):

            gp, loss = env.fit_gp(t)
            pred = gp.posterior(torch.tensor(X, dtype=torch.double)).mean # gp(torch.tensor(X)).mean.T
            partitioning = FastNondominatedPartitioning(
                ref_point=torch.tensor(env.min_function_values, dtype=torch.double),
                Y=pred)
            
            if args.learner == "qEHVI":
                learner = qExpectedHypervolumeImprovement(
					model=gp,
					ref_point=torch.tensor(env.min_function_values, dtype=torch.double),
					partitioning=partitioning,
				)
            elif args.learner == "qNEHVI":
                learner = qNoisyExpectedHypervolumeImprovement(
					model=gp,
					ref_point=torch.tensor(env.min_function_values, dtype=torch.double),
					X_baseline=torch.tensor(env.history['x'], dtype=torch.double),
					prune_baseline=True,
				)
            elif args.learner == "qHVKG":
                learner = qHypervolumeKnowledgeGradient(
					model=gp,
					ref_point=torch.tensor(env.min_function_values, dtype=torch.double),
					num_fantasies=1,
					num_pareto=1,
				)
            elif args.learner == "qParEGO":
                 pred = gp.posterior(torch.tensor(env.history['x'], dtype=torch.double)).mean
                 weights = sample_simplex(d = args.f_num, dtype=torch.double).squeeze()
                 objective = GenericMCObjective(get_chebyshev_scalarization(weights=weights, Y=pred))
                 learner = qNoisyExpectedImprovement(
					model=gp,
					objective=objective,
					X_baseline=torch.tensor(np.array(env.history['x']), dtype=torch.double),
				)
            elif args.learner == "JES":
                pareto_sets, pareto_fronts = JES_utils.sample_pareto_sets_and_fronts(
					model=gp,
					num_pareto_samples=1,
					num_pareto_points=1,
					bounds=standard_bounds,
				)
                hypercell_bounds = JES_utils.compute_box_decomposition(pareto_fronts)
                learner = qLowerBoundMultiObjectiveJointEntropySearch(
					model=gp,
					pareto_sets=pareto_sets.squeeze(1),
					pareto_fronts=pareto_fronts.squeeze(1),
					hypercell_bounds=hypercell_bounds.squeeze(1),
				)

            if args.discrete:
                x_best = X[int(learner(torch.tensor(X).unsqueeze(1)).argmax(dim=0).detach().numpy())]
            else:
                if args.learner == "qParEGO" and t < args.ls_learned_freq:
                    x_best = np.random.rand(env.domain_dim)
                else:
                    candidates, _ = optimize_acqf(
                        acq_function=learner,
                        bounds=standard_bounds.double(),
                        q=1,
                        num_restarts=1,
                        raw_samples=1,
                        options={"batch_limit": 1, "maxiter": 10},
                        sequential=True,
                    )
                    x_best = candidates.detach().numpy()[0]
            y_star, reward, regret = env.step(x_best)

        print(f"E: {e} | F: {args.function_type} | P: {args.learner} | R: {regret}")
        if args.function_type == "YAHPO":
            filename = '{}_function_type_{}_{}_online_ls_{}_per_{}_obs_{}_dis_{}_episode_{}.pkl'.format(args.learner, args.function_type, args.yahpo_scenario, args.online_ls, args.perturb_noise_level, args.observation_noise_level, args.discrete, e)
        elif args.function_type == "NERF_synthetic" or args.function_type == "NERF_real" or args.function_type == "NERF_synthetic_fnum_3":
            filename = '{}_function_type_{}_{}_per_{}_obs_{}_dis_{}_episode_{}.pkl'.format(args.learner, args.function_type, args.NERF_scene, args.perturb_noise_level, args.observation_noise_level, args.discrete, e)
        else:
            filename = '{}_function_type_{}_dis_{}_episode_{}.pkl'.format(args.learner, args.function_type, args.discrete,  e)
        with open(os.path.join(f'./results/', filename), 'wb') as f:
            pickle.dump(env.history, f)