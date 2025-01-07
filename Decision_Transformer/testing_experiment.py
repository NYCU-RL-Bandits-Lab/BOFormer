import numpy as np
import torch
import argparse
import pickle
import random
import sys
import os
from decision_transformer.models.decision_transformer import DecisionTransformer
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
from Environment.env import Environment, construct_state_action_pair
from Environment.function_preprocessing import domain
from Environment.env import Environment
from tqdm import tqdm
torch.cuda.set_device(0)   
import warnings
warnings.filterwarnings("ignore")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def test(variant, args):
    # load dataset
    mode = variant.get('mode', 'normal')
    env_name = f'BO_{variant["f_num"]}'

    state_dim = args.domain_size*(2*variant['f_num']+variant['f_num']+1)
    act_dim = 1

    K = variant['K']
    max_ep_len = args.T
    num_eval_episodes = variant['num_eval_episodes']

    model = DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=100,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4*variant['embed_dim'],
            activation_function=variant['activation_function'],
            action_tanh=False,
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
        )
    
    for train_ep in [9]:
        target_return = 0
        pretrain_path = 'Decision_Transformer/preTrained/{}/{}.pth'.format(env_name, train_ep)
        model.load_state_dict(torch.load(pretrain_path))
        model.to(device)
        print('Success load model from {}'.format(pretrain_path))

        test_f = [variant['function_type']]

        env = Environment(T = args.T, 
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
                      domain_dim = 1,
                      NERF_scene = args.NERF_scene)
        regrets = []
        print('Test on function {}'.format(test_f))
        for e in tqdm(range(num_eval_episodes)):
            seed=args.seed+e*10
            env.reset(seed=seed)
            # initial sample
            X = domain(args.function_type, args.domain_size, seed, d = 1) 
            if X.shape[0] > args.domain_size:
                X = X[:args.domain_size]
            y_star, reward, regret = env.step(X[random.randint(0,min(np.shape(X)[0]-1,args.domain_size-1))])
            gp, _ = env.fit_gp(0)
            state = construct_state_action_pair(X, gp, y_star, 0)
            # we keep all the histories on the device
            # note that the latest action and reward will be "padding"
            states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
            actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
            rewards = torch.zeros(0, device=device, dtype=torch.float32)
            ep_regret = []

            ep_return = target_return
            target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
            timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

            episode_return, episode_length = 0, 0
            for t in range(max_ep_len):

                # add padding
                actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
                rewards = torch.cat([rewards, torch.zeros(1, device=device)])

                action = model.get_action(
                    states.to(dtype=torch.float32),
                    actions.to(dtype=torch.float32),
                    rewards.to(dtype=torch.float32),
                    target_return.to(dtype=torch.float32),
                    timesteps.to(dtype=torch.long),
                )
                actions[-1] = action
                action = action.detach().cpu().numpy()
                #print(action)
                y_star, reward, regret = env.step(X[int(action)])

                gp, _ = env.fit_gp(t)

                X = domain(args.function_type, args.domain_size, seed, d = 1)
                if X.shape[0] > args.domain_size:
                    X = X[:args.domain_size]
                state = construct_state_action_pair(X, gp, y_star, 0)
                #print(action)
                ep_regret.append(regret)
                # state = state[env.index(action)].astype('float32') # for BO dataset
                cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
                states = torch.cat([states, cur_state], dim=0)
                rewards[-1] = reward

                episode_return += reward
                episode_length += 1

            regrets.append(ep_regret)
            os.makedirs('./results', exist_ok=True)
            if args.function_type == "YAHPO":
                filename = '{}_function_type_{}_{}_online_ls_{}_per_{}_obs_{}_dis_{}_episode_{}.pkl'.format("DT", args.function_type, args.yahpo_scenario, args.online_ls, args.perturb_noise_level, args.observation_noise_level, args.discrete, e)
            elif args.function_type == "NERF_synthetic" or args.function_type == "NERF_real" or args.function_type == "NERF_synthetic_fnum_3":
                filename = '{}_function_type_{}_{}_per_{}_obs_{}_dis_{}_episode_{}.pkl'.format("DT", args.function_type, args.NERF_scene, args.perturb_noise_level, args.observation_noise_level, args.discrete,e)
            else:
                filename = '{}_function_type_{}_dis_{}_episode_{}.pkl'.format("DT", args.function_type, e)
            with open(os.path.join(f'./results/DT', filename), 'wb') as f:
                pickle.dump(env.history, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='BO')
    parser.add_argument('--dataset', type=str, default='medium')
    parser.add_argument('--T', type=int, default=30, help='total iteration')  
    parser.add_argument('--mode', type=str, default='normal')
    parser.add_argument('--K', type=int, default=100)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model_type', type=str, default='dt')  # dt for decision transformer, bc for behavior cloning
    parser.add_argument('--embed_dim', type=int, default=500)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10)
    parser.add_argument('--num_eval_episodes', type=int, default=100)
    parser.add_argument('--max_iters', type=int, default=20)
    parser.add_argument('--num_steps_per_iter', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=False)
    parser.add_argument('--f_num', type=int, default=2)
    parser.add_argument('--function_type', type=str, default="AR")
    parser.add_argument('--NERF_scene', type=str, default="drums", help='NERF_scene')
    parser.add_argument('--yahpo_scenario', type=str, default='lcbench', help='lcbench, rbv2_xgboost, rbv2_svm, rbv2_glmnet')
    parser.add_argument('--perturb_noise_level', type=float, default=0.01, help='perturbed noise')
    parser.add_argument('--observation_noise_level', type=float, default=0.0, help='observation noise')
    parser.add_argument('--domain_size', type=int, default=1000)
    parser.add_argument('--ls_learned_freq', type=int, default=10)
    parser.add_argument('--online_ls', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--discrete', type=int, default=1)
    
    args = parser.parse_args()

    test(variant=vars(args), args=args)