import pickle
import argparse
import argparse
import random
import numpy as np
import torch
import transformers
import tqdm
from modeling_gpt2 import GPT2Model
import sys, os
import Environment.env
import Environment.function_preprocessing
import Environment.benchmark_functions
import os
# from torch.utils.tensorboard import SummaryWriter
from torch import nn
from scipy.stats import rankdata
import copy
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from botorch.acquisition import AcquisitionFunction
from scipy.stats import rankdata
import sobol_seq
import warnings
warnings.filterwarnings("ignore")

class QT(torch.nn.Module):
    def __init__(self, 
                 T, 
                 domain_size, 
                 f_num, 
                 gamma = 0.98, 
                 hidden_size = 128, 
                 lr = 0.01, 
                 weight_decay = 0.1, 
                 n_layer = 4, 
                 n_head = 4, 
                 n_batch = 1,
                 n_positions = 301,
                 warmup_steps=1000,
                 update_freq = 1, 
                 optimizer = "sgd",
                 dropout = 0.1,
                 epsilon = 0.1,
                 target_update_freq = 1,
                 batch_size = 16,
                 initial_sample = 1,
                 temperature = 1000,
                 device = "cpu"):
        super(QT, self).__init__()

        self.device = device
        self.update_freq = update_freq
        self.hidden_size = hidden_size
        self.lr = lr
        self.weight_decay = weight_decay
        
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            n_layer=n_layer,
            n_head=n_head,
            n_inner=4*hidden_size,
            activation_function='relu',
            n_positions=n_positions,
            resid_pdrop=dropout,
            attn_pdrop=dropout,
            embd_pdrop=dropout
        )
        self.n_positions = n_positions
        self.n_batch = n_batch
        self.epsilon = epsilon # for exploration
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.initial_sample = initial_sample
        self.temperature = temperature
        self.max_length = T
        self.domain_size = domain_size
        self.transformer = GPT2Model(config)
        self.state_action_dim = f_num * 3 + 1
        self.gamma = gamma
        self.embed_reward = torch.nn.Linear(1, hidden_size)
        self.embed_q_value = torch.nn.Linear(1, hidden_size)
        self.embed_time = torch.nn.Linear(1, hidden_size)
        self.embed_state_action = torch.nn.Linear(f_num * 3 + 1, hidden_size)

        self.predict_state_action = torch.nn.Linear(hidden_size, self.state_action_dim)
        self.predict_q_value = torch.nn.Linear(hidden_size, 1)
        self.predict_reward = torch.nn.Linear(hidden_size, 1)

        self.trajectory_buffer = []
        self.trajectory = []
        self.actions = []
        

    def forward_without_q(self, state_actions, rewards, next_state_actions):
        q_values = []
        pred_len = state_actions.size()[1]
        time_embeddings = self.embed_time(torch.from_numpy(np.arange(pred_len)).float().reshape(1,pred_len,1).to(self.device))
        state_action_embeddings = self.embed_state_action(state_actions.to(self.device)) + time_embeddings
        reward_embeddings = self.embed_reward(rewards.to(self.device)) + time_embeddings
        for i in range(pred_len):
            if i == 0:
                transformer_outputs = self.transformer(inputs_embeds=state_action_embeddings[:,:i+1,:])
                x = transformer_outputs['last_hidden_state']
                q_values.append(self.predict_q_value(x).detach())
            else:
                q_value_embeddings = self.embed_q_value(torch.cat(q_values, axis = 1)) + time_embeddings[:,:i,:]# (sequence_length - 1, hidden_size)
                inputs = torch.stack((state_action_embeddings[:,:i,:], q_value_embeddings, reward_embeddings[:,:i,:]), dim=2).reshape(state_action_embeddings.size(0), i*3, self.hidden_size)
                # this makes the sequence look like ((s_1, a_1), r_1, Q_1, (s_2, a_2), ...)
                inputs = torch.cat((inputs, state_action_embeddings[:,i,:].unsqueeze(1)), dim=1) # (3 * sequence_length - 2, hidden_size)
                # feed in the input embeddings (not word indices as in NLP) to the model
                
                # fit the n_position size
                if inputs.size()[1] > self.n_positions:
                    inputs = inputs[:,-self.n_positions:,:]

                transformer_outputs = self.transformer(inputs_embeds=inputs)
                x = transformer_outputs['last_hidden_state'] # (3 * sequence_length - 2, hidden_size)

                # get predictions
                q_value_pred = self.predict_q_value(torch.index_select(x.cpu(), 1, torch.arange(3,x.size(1),3,dtype=int)).to(self.device)) 
                q_values.append(q_value_pred[:,-1:,:].detach())
        
        q_values = torch.cat(q_values, axis = 1)
        if next_state_actions is None:
            return q_values
        
        # next_state_actions, (batch_size, pred_len, domain_size, state_action_dim)
        candidate_q_values = []
        pred_len = state_actions.size()[1]
        for i in range(pred_len):
            temp = []
            candidate_points = random.sample(range(self.domain_size), int(self.domain_size*self.sample_rate))
            if i != 0:
                q_value_embeddings = self.embed_q_value(torch.cat(candidate_q_values, axis = 1)) + time_embeddings[:,:i,:] # (sequence_length - 1, hidden_size)
                
            for j in candidate_points:
                temp_state_actions = torch.from_numpy(np.array(next_state_actions))[:,i:i+1,j,:].float()
                state_action_embeddings_new = self.embed_state_action(temp_state_actions.to(self.device)) + time_embeddings[:,i:i+1,:]
                if i == 0:
                    state_action_embeddings_new = torch.cat((state_action_embeddings[:,:i,:], state_action_embeddings_new), dim = 1)
                    transformer_outputs = self.transformer(inputs_embeds=state_action_embeddings_new)
                    x = transformer_outputs['last_hidden_state']
                    temp.append(self.predict_q_value(x).detach())
                else:
                    
                    inputs = torch.stack((state_action_embeddings[:,:i,:], q_value_embeddings, reward_embeddings[:,:i,:]), dim=2).reshape(state_action_embeddings.size(0), i*3, self.hidden_size)
                    # this makes the sequence look like ((s_1, a_1), r_1, Q_1, (s_2, a_2), ...)
                    inputs = torch.cat((inputs, state_action_embeddings_new), dim=1) # (3 * sequence_length - 2, hidden_size)
                    # feed in the input embeddings (not word indices as in NLP) to the model

                    # fit the n_position size
                    if inputs.size()[1] > self.n_positions:
                        inputs = inputs[:,-self.n_positions:,:]

                    transformer_outputs = self.transformer(inputs_embeds=inputs)
                    x = transformer_outputs['last_hidden_state'] # (3 * sequence_length - 2, hidden_size)

                    # get predictions
                    q_value_pred = self.predict_q_value(torch.index_select(x.cpu(), 1, torch.arange(3,x.size(1),3,dtype=int)).to(self.device)) 
                    temp.append(q_value_pred[:,-1:,:].detach())
            candidate_q_values.append(torch.max(torch.stack(temp),dim = 0)[0])
        candidate_q_values = torch.cat(candidate_q_values, axis = 1)
        return torch.max(torch.cat((candidate_q_values, q_values), axis=2), axis = 2)[0].unsqueeze(2)

    def forward(self, state_actions, rewards, q_values):

        if rewards == None or q_values == None:
            state_actions = torch.from_numpy(state_actions).float().unsqueeze(0).unsqueeze(0).to(self.device)
            state_action_embeddings = self.embed_state_action(state_actions)
            transformer_outputs = self.transformer(inputs_embeds=state_action_embeddings)
            x = transformer_outputs['last_hidden_state']
            q_value_pred = self.predict_q_value(x)  
            return q_value_pred
        
        batch_size = state_actions.size(0)
        pred_len = state_actions.size(1)

        # embed each modality with a different head
        time_embeddings = self.embed_time(torch.from_numpy(np.arange(pred_len)).float().unsqueeze(1).to(self.device)).unsqueeze(0) # (sequence_length + 1, hidden_size)
        state_action_embeddings = self.embed_state_action(state_actions) + torch.tile(time_embeddings, (batch_size,1,1)) # (sequence_length, hidden_size)
        q_value_embeddings = self.embed_q_value(q_values) + torch.tile(time_embeddings[:,:pred_len-1,:], (batch_size,1,1))# (sequence_length - 1, hidden_size)
        reward_embeddings = self.embed_reward(rewards) + torch.tile(time_embeddings[:,:pred_len-1,:], (batch_size,1,1))# (sequence_length - 1, hidden_size)
        
        inputs = torch.stack((state_action_embeddings[:,:pred_len-1,:], q_value_embeddings, reward_embeddings), dim=2).reshape(state_action_embeddings.size(0), (pred_len-1)*3, self.hidden_size)
        # this makes the sequence look like ((s_1, a_1), r_1, Q_1, (s_2, a_2), ...)
        inputs = torch.cat((inputs, state_action_embeddings[:,-1:,:]), dim=1) # (3 * sequence_length - 2, hidden_size)
        # feed in the input embeddings (not word indices as in NLP) to the model
        
        # fit the n_position size
        if inputs.size()[1] > self.n_positions:
            inputs = inputs[:,-self.n_positions:,:]

        transformer_outputs = self.transformer(inputs_embeds=inputs)
        x = transformer_outputs['last_hidden_state'] # (3 * sequence_length - 2, hidden_size)

        # get predictions
        q_value_pred = self.predict_q_value(torch.index_select(x.cpu(), 1, torch.arange(0,x.size(1),3,dtype=int)).to(self.device)) 

        return q_value_pred
    
    def select_action(self, state_actions, target_network, N_m = 1):
        torch.cuda.empty_cache()
        # first action is random
        if len(self.trajectory) < self.initial_sample:
            action = random.randint(0, self.domain_size - 1)
            return action
        
        previous_state_actions = torch.from_numpy(np.array([b["state_action"] for b in self.trajectory]).reshape(1,len(self.trajectory),self.state_action_dim)).float()
        previous_rewards = torch.from_numpy(np.array([b["reward"] for b in self.trajectory]).reshape(1,len(self.trajectory),1)).float()
        self.previous_q_values = target_network.forward_without_q(previous_state_actions, previous_rewards, None).detach()
        state_actions = torch.from_numpy(state_actions).float()
    
        batch_state_actions = torch.cat((torch.tile(previous_state_actions,(self.domain_size,1,1)), state_actions.unsqueeze(1)), axis=1).to(self.device)
        # (domain_size, sequence_length, state_action_dim)
        batch_rewards = torch.tile(previous_rewards, (self.domain_size,1,1)).to(self.device)
        # (domain_size, sequence_length - 1, 1)
        batch_q_values = torch.tile(self.previous_q_values, (self.domain_size,1,1))
        # (domain_size, sequence_length - 1, 1)
        q_values = self.forward(batch_state_actions, batch_rewards, batch_q_values).detach()

        # select the best action based on the q_values
        dist = torch.distributions.Categorical(logits = q_values[:,-1,:].cpu().squeeze().double()*self.temperature)
        action = dist.sample()
        self.actions.append(action)
        # action = np.argmax(q_values[:,-1,:].cpu())
        return [action]

    def reset_trajectory(self):
        self.actions = []
        self.trajectory = []
        # if self.device != 'cpu': 
        #     torch.cuda.empty_cache()

def args_to_info(args):
    learner_info = "{}_{}_domain_{}_fnum_{}_gamma_{}_hidden_{}_lr_{}_weight_decay_{}_seed_{}_n_layer_{}_n_head_{}_n_positions_{}_dropout_{}_epsilon_{}_target_update_freq_{}_batch_size_{}_demo_rate_{}_buffer_size_{}_new_reward_{}_sample_rate_{}".format(
        args.model_episode,
        args.model_type,
        args.domain_size,  
        args.f_num, 
        args.gamma,
        args.hidden_size,
        args.lr,
        args.weight_decay,
        args.seed,
        args.n_layer,
        args.n_head,
        args.n_positions,
        args.dropout,
        args.epsilon,
        args.target_update_freq,
        args.batch_size,
        args.demo_rate,
        args.buffer_size,
        args.new_reward,
        args.sample_rate
        )
    return learner_info

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Multi-objective BO')
    parser.add_argument('--device', type=str, default="0", help='gpu device')
    parser.add_argument('--env', type=str, default='BO')
    parser.add_argument('--testing_episode', type=int, default=100)
    parser.add_argument('--T', type=int, default=100)
    parser.add_argument('--domain_size', type=int, default=1000)
    parser.add_argument('--f_num', type=int, default=2)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--weight_decay', type=float, default=0.00001)
    parser.add_argument('--n_batch', type=int, default=1)
    parser.add_argument('--update_freq', type=int, default=1)
    parser.add_argument('--n_layer', type=int, default=4)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--target_update_freq', type=int, default=5)
    parser.add_argument('--n_positions', type=int, default=31)
    parser.add_argument('--warmup_steps', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--function_type', type=str, default='NERF_synthetic')
    parser.add_argument('--model_episode', type=int, default=1000)
    parser.add_argument('--model_type', type=str, default="train")
    parser.add_argument('--optimizer', type=str, default="adam")
    parser.add_argument('--dropout', type=float, default = 0.1)
    parser.add_argument('--epsilon', type=float, default = 0.1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--sample_rate', type=float, default=0.005)
    parser.add_argument('--demo_rate', type=float, default=0.01)
    parser.add_argument('--buffer_size', type=int, default=128)
    parser.add_argument('--new_reward',type=int, default=1)
    parser.add_argument('--ls_learned_freq', type=int, default=10, help='freq of learning ls')

    parser.add_argument('--perturb_noise_level', type=float, default=0.1, help='perturbed noise')
    parser.add_argument('--observation_noise_level', type=float, default=0.1, help='observation noise')

    parser.add_argument('--update_step', type=int, default=10, help='# of GD steps')
    parser.add_argument('--N_m', type=int, default=1, help='N_m')
    parser.add_argument('--N_local', type=int, default=1, help='N_local')
    parser.add_argument('--initial_sample', type=int, default=1, help='# of initial sample')
    parser.add_argument('--online_ls', type=int, default=1, help='ls in testing')
    parser.add_argument('--temperature', type=int, default=1000, help='temperature of softmax policy')
    parser.add_argument('--domain_dim', type=int, default=2, help='domain dimension')
    parser.add_argument('--yahpo_scenario', type=str, default='lcbench', help='lcbench, rbv2_xgboost, rbv2_svm, rbv2_glmnet')
    parser.add_argument('--NERF_scene', type=str, default="drums", help='NERF_scene')
    parser.add_argument('--discrete', type=int, default=1, help='discrete')
    parser.add_argument('--record_idx', type=int, default=-1)
    args = parser.parse_args()
    learner_info = args_to_info(args)
    print(args.function_type, learner_info)
    
    if "RBF" in args.function_type or "matern" in args.function_type:
        args.domain_dim = 1

    record_index = -1
    with open('BOFormer_record.txt', 'r') as f:
        for line in f.readlines():
            index, namespace_str = line.split('\t', 1)
            args_str = namespace_str[10:-2]  # remove 'Namespace(' and ')'
            args_list = args_str.split(', ')
            args_dict = {arg.split('=')[0]: arg.split('=')[1] for arg in args_list}
            args_dict['model_episode'] = int(args.model_episode)
            args_dict['model_type'] = args_dict['function_type'][1:-1]
            args_dict = argparse.Namespace(**args_dict)
            c_info = args_to_info(args_dict)
            print(c_info)
            if c_info == learner_info:
                record_index = int(index)
                break
    
    if args.record_idx != -1:
        record_index = args.record_idx

    print("record index: ", record_index)
    if args.device != "cpu":
        device = torch.device("cuda:" + args.device if torch.cuda.is_available() else "cpu")
    else: 
        device = "cpu"
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)

    # set seed for reproduc
    if args.seed > 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    env = Environment.env.Environment(T = args.T, 
                      domain_size = args.domain_size, 
                      f_num = args.f_num, 
                      function_type = args.function_type,
                      yahpo_scenario= args.yahpo_scenario, 
                      seed = args.seed,
                      new_reward = args.new_reward,
                      perturb_noise_level = args.perturb_noise_level,
                      observation_noise_level = args.observation_noise_level,
                      ls_learned_freq = args.ls_learned_freq,
                      online_ls=args.online_ls,
                      domain_dim = args.domain_dim,
                      NERF_scene = args.NERF_scene)
    
    args.domain_size = min(np.shape(env.X)[0], args.domain_size)

    if args.function_type == "NERF_synthetic":
       args.domain_size = np.shape(env.X)[0]
    
    learner = QT(T = args.T, 
                 domain_size = args.domain_size, 
                 f_num = args.f_num, 
                 gamma = args.gamma, 
                 hidden_size = args.hidden_size, 
                 lr = args.lr, 
                 weight_decay = args.weight_decay, 
                 n_layer = args.n_layer, 
                 n_head = args.n_head,
                 n_batch = args.n_batch,
                 n_positions = args.n_positions,
                 warmup_steps = args.warmup_steps,
                 update_freq = args.update_freq,
                 optimizer = args.optimizer,
                 dropout = args.dropout,
                 epsilon = args.epsilon,
                 target_update_freq = args.target_update_freq,
                 batch_size = args.batch_size,
                 initial_sample = args.initial_sample,
                 temperature = args.temperature,
                 device = device).to(device)
     
    if args.model_episode > 0:
        learner.load_state_dict(torch.load("./BOFormer_models/model{}/{}.pth".format(record_index, args.model_episode), map_location=device))
    learner.eval()

    target_network = QT(T = args.T, 
                 domain_size = args.domain_size, 
                 f_num = args.f_num, 
                 gamma = args.gamma, 
                 hidden_size = args.hidden_size, 
                 lr = args.lr, 
                 weight_decay = args.weight_decay, 
                 n_layer = args.n_layer, 
                 n_head = args.n_head,
                 n_batch = args.n_batch,
                 n_positions = args.n_positions,
                 warmup_steps = args.warmup_steps,
                 update_freq = args.update_freq,
                 optimizer = args.optimizer,
                 dropout = args.dropout,
                 epsilon = args.epsilon,
                 target_update_freq = args.target_update_freq,
                 batch_size = args.batch_size,
                 temperature = args.temperature,
                 device = device).to(device)
    
    target_network.load_state_dict(learner.state_dict())
    target_network.eval()

    os.makedirs(f'./BOFormer_testings/model{record_index}/', exist_ok=True)
    with open(f'./BOFormer_testings/model{record_index}/args.txt', 'a') as f:
        f.write(str(args)+'\n')
    
    for e in range(args.testing_episode):
        # initialization
        seed=args.seed+e*10
        env.reset(seed=seed, episode=e)
        env.history['info'] = str(args)
        sas = [] # record

        # initial sample
        X = Environment.function_preprocessing.domain(args.function_type, args.domain_size, seed, args.domain_dim, discrete=args.discrete)
        y_star, reward, regret = env.step(X[random.randint(0,args.domain_size-1)])
        gp, _ = env.fit_gp(0)
        state_actions = Environment.env.construct_state_action_pair(X, gp, y_star, 0)

        # record transition
        learner.trajectory.append({"state_action": np.array([0.0]*(3*args.f_num+1)), "reward": float(reward), "next_state_actions": state_actions})
        sas.append(state_actions)

        # training iterations
        for t in tqdm.tqdm(range(1, args.T)):            
            
            # select action
            actions = learner.select_action(state_actions, target_network, N_m  = args.N_m)
            if len(actions) == 1:
                action = actions[0]
                env.history["actions"].append(action)
                # env update
                y_star, reward, regret = env.step(X[action])
            else:
                candidate = []
                for action in actions:
                    candidate.append(X[action] + 0.05*(sobol_seq.i4_sobol_generate(env.domain_dim, args.N_local, seed+t)-0.5))
                candidate = np.clip(np.concatenate(candidate, axis = 0), 0, 1)
                state_action_pairs = Environment.env.construct_state_action_pair(candidate, gp, y_star, t/args.T)
                action = learner.select_action(state_action_pairs, target_network, N_m = 1)[0]
                # env update
                y_star, reward, regret = env.step(candidate[action])
            # learn ls for GP
            gp, _ = env.fit_gp(t)
                
            X = Environment.function_preprocessing.domain(args.function_type, args.domain_size, seed+t, args.domain_dim, discrete=args.discrete) 
            next_state_actions = Environment.env.construct_state_action_pair(X, gp, y_star, t/args.T)

            # record transition
            learner.trajectory.append({"state_action": state_actions[action], "reward": reward})
            
            # update current state
            state_actions = next_state_actions
            sas.append(state_actions)

        # record final transition
        learner.trajectory.append({"state_action": state_actions[action], "reward": 0.0})

        # learner update
        learner.trajectory_buffer.append(learner.trajectory)
        # loss = learner.update(target_network)
        learner.reset_trajectory()
        # env.history["sa"] = sas
        print('EP:{} | R: {:.3f} | scene: {}'.format(e, regret, env.episode))
        if args.function_type == "YAHPO":
            filename = '{}_model_{}_function_type_{}_{}_dim_{}_N_m_{}_N_local_{}_ls_learned_freq_{}_initial_sample_{}_online_ls_{}_episode_{}.pkl'.format(
            "BOFormer", 
            args.model_episode,
            args.function_type,
            args.yahpo_scenario,
            args.domain_dim,
            args.N_m,
            args.N_local,
            args.ls_learned_freq,
            args.initial_sample,
            args.online_ls,
            e)
        elif args.function_type == "NERF_synthetic" or args.function_type == "NERF_real" or args.function_type == "NERF_synthetic_fnum_3":
            filename = '{}_model_{}_function_type_{}_{}_dim_{}_N_m_{}_N_local_{}_ls_learned_freq_{}_initial_sample_{}_online_ls_{}_episode_{}.pkl'.format(
            "BOFormer", 
            args.model_episode,
            args.function_type,
            args.NERF_scene,
            args.domain_dim,
            args.N_m,
            args.N_local,
            args.ls_learned_freq,
            args.initial_sample,
            args.online_ls,
            e)
        else:
            filename = '{}_model_{}_function_type_{}_dim_{}_N_m_{}_N_local_{}_ls_learned_freq_{}_initial_sample_{}_online_ls_{}_episode_{}.pkl'.format(
            "BOFormer", 
            args.model_episode,
            args.function_type,
            args.domain_dim,
            args.N_m,
            args.N_local,
            args.ls_learned_freq,
            args.initial_sample,
            args.online_ls,
            e)

        # save the history
        with open(os.path.join(f'./BOFormer_testings/model{record_index}/', filename), 'wb') as f:
            pickle.dump(env.history, f)
