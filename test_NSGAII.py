import numpy as np
import argparse
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.sampling.lhs import LatinHypercubeSampling
from pymoo.factory import (
    get_crossover,
    get_mutation,
    get_sampling,
    get_termination,
)
import sys, os
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import Environment.env
from tqdm import tqdm
import pickle
import time

class Fun(Problem):
    def __init__(self, d, f_num, F:list):
        super().__init__(n_var=d, n_obj=f_num, n_constr=0, xl=np.array([0 for _ in range(d)]), xu=np.array([1 for _ in range(d)]))
        self.F = F # list of functions

    def _evaluate(self, x, out ,*args, **kwargs):
        out["F"] = np.column_stack([[1-self.F[i](x[j]) for j in range(len(x))] for i in range(len(self.F))])

if __name__=='__main__':
    os.makedirs(f"results/nsga2/", exist_ok=True)
    parser = argparse.ArgumentParser(description='Multi-objective BO')
    parser.add_argument('--function_type', type=str, default="AR", help='RBF_0.05, RBF_0.2, RBF_0.3, matern52_0.05, matern52_0.2, matern52_0.3, BC, AR, ARS, DRZ, Branin, Currin, YAHPO')
    parser.add_argument('--yahpo_scenario', type=str, default='lcbench', help='lcbench, rbv2_xgboost, rbv2_svm, rbv2_glmnet')
    parser.add_argument('--NERF_scene', type=str, default="lego", help='NERF_scene')
    parser.add_argument('--domain_size', type=int, default=1000, help='domain size')
    parser.add_argument('--f_num', type=int, default=2, help='number of objective function')
    parser.add_argument('--T', type=int, default=100, help='total iteration')
    parser.add_argument('--episode', type=int, default=100, help='number of episodes')
    parser.add_argument('--ls_learned_freq', type=int, default=10, help='freq of learning ls')
    parser.add_argument('--learner', type=str, default='NSGA2', help='qEHVI, qNEHVI, qHVKG, qParEGO, JES')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--perturb_noise_level', type=float, default=0.1, help='perturbed noise')
    parser.add_argument('--observation_noise_level', type=float, default=0.1, help='observation noise')
    parser.add_argument('--domain_dim', type=int, default=2, help='domain dimension')
    parser.add_argument('--discrete', type=int, default=0, help='discrete')
    args = parser.parse_args()

    env = Environment.env.Environment(T = args.T, 
                      domain_size = args.domain_size, 
                      f_num = args.f_num, 
                      function_type = args.function_type,
                      yahpo_scenario= args.yahpo_scenario,
                      NERF_scene = args.NERF_scene,
                      seed = args.seed,
                      perturb_noise_level = args.perturb_noise_level,
                      observation_noise_level = args.observation_noise_level,
                      new_reward = 0,
                      domain_dim = args.domain_dim,
                      discrete= args.discrete)

    for e in range(args.episode):
        seed=args.seed+e*10
        env.reset(seed=seed, episode=e)
        env.history['info'] = str(args)
        regrets = []
        losses = []
        actions_record = []

        F = env.F
        function_type = args.function_type
        if (function_type == "train" or function_type == "RBF_0.05" or function_type == "RBF_0.2"  or function_type == "RBF_0.3" or function_type == "matern52_0.05" or function_type == "matern52_0.2" or function_type == "matern52_0.3"):
            d = 1
        elif (function_type == "RE24" or function_type == "DR" or function_type == "BC" or function_type == "ARS" or function_type == "DRZ" or function_type == "Branin" or function_type == "Currin" or function_type == "AR" or function_type == "DR" or function_type == "ARa" or function_type == "BCD" or function_type == "ASR" or function_type == "ABC" or function_type == "ACD" or function_type == "ABD"):
            d = 2
        elif (function_type == "RE25" or function_type == "RE22" or function_type == "RE31"):
            d = 3
        elif (function_type == "RE33" or function_type == "RE21" or function_type == "RE23" or function_type == "RE32" ):
            d = 4
        elif (function_type == "RE34" or function_type == "NERF_synthetic"):
            d = 5
        elif (function_type == "RE35"):
            d = 7
        elif (function_type == "HPO"):
            d = 6
        elif (function_type == "YAHPO"):
            if args.yahpo_scenario == "lcbench":
                d = 4
            elif args.yahpo_scenario == "rbv2_xgboost":
                d = 9
            else:
                d = 3
        problem = Fun(d, args.f_num, F)
        solver = NSGA2(pop_size=10, sampling = get_sampling("real_random") ,eliminate_duplicates=True)
        res = minimize(problem, solver, ('n_gen', 10), save_history=True)
        X_history = np.array([algo.pop.get('X') for algo in res.history])
        Y_history = np.array([algo.pop.get('F') for algo in res.history])
        X_history = X_history.reshape((args.T, d))
        Y_history = Y_history.reshape((args.T, args.f_num))
        regrets = []
        
        for i in tqdm(range(args.T)):
            x = X_history[i]
            y_star, reward, regret = env.step(x)

        # print(f"E: {e} | F: {args.function_type} | R: {regret}")
        if args.function_type == "YAHPO":
            filename = '{}_function_type_{}_{}_per_{}_obs_{}_episode_{}.pkl'.format(args.learner, args.function_type, args.yahpo_scenario, args.perturb_noise_level, args.observation_noise_level, e)
        elif args.function_type.startswith("NERF"):
            filename = '{}_function_type_{}_{}_per_{}_obs_{}_episode_{}.pkl'.format(args.learner, args.function_type, args.NERF_scene, args.perturb_noise_level, args.observation_noise_level, e)
        else:
            filename = '{}_function_type_{}_episode_{}.pkl'.format(args.learner, args.function_type, e)
        with open(os.path.join(f'./results/nsga2/', filename), 'wb') as f:
            pickle.dump(env.history, f)
        