import pickle
import argparse
import argparse
import random
import numpy as np
import torch
import transformers
import tqdm
import math
from modeling_gpt2 import GPT2Model
import os
import Environment.env
import Environment.function_preprocessing
import pathlib
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from ehi import ExpectedHypervolumeImprovement
from prioritized_memory import Memory
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement, qNoisyExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.joint_entropy_search import qLowerBoundMultiObjectiveJointEntropySearch
from botorch.optim.optimize import optimize_acqf

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
                 sample_rate = 0.1,
                 buffer_size = 64,
                 initial_sample = 1,
                 temperature = 1000,
                 bc_rate = 0.0,
                 device = "cpu"):
        super(QT, self).__init__()
        self.initial_sample = initial_sample
        self.temperature = temperature
        self.bc_rate = bc_rate
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
        self.buffer_size = buffer_size
        self.sample_rate = sample_rate
        self.n_positions = n_positions
        self.n_batch = n_batch
        self.epsilon = epsilon # for exploration
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
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

        self.trajectory_buffer = Memory(buffer_size)
        self.trajectory = []

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=10.0, std=0.02/math.sqrt(2 * n_layer))
        # report number of parameters
        print("number of parameters: %.2fM" % (sum(p.numel() for p in self.parameters())/1e6,))

        if optimizer == "sgd":
            self.optimizer = torch.optim.SGD(params=self.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        elif optimizer == "adam":
            self.optimizer = torch.optim.AdamW(params=self.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError("Unknown optimizer %s" % optimizer)
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,lambda steps: min((steps+1)/warmup_steps, 1))
        

    def forward_without_q(self, state_actions, rewards, next_state_actions):
        q_values = []
        pred_len = state_actions.size()[1]
        time_embeddings = self.embed_time(torch.from_numpy(np.arange(pred_len)).float().reshape(1,pred_len,1).to(self.device))
        state_action_embeddings = self.embed_state_action(state_actions.to(self.device)) + time_embeddings
        for i in range(pred_len):
            if i == 0:
                transformer_outputs = self.transformer(inputs_embeds=state_action_embeddings[:,:i+1,:])
                x = transformer_outputs['last_hidden_state']
                q_values.append(self.predict_q_value(x).detach())
            else:
                inputs = state_action_embeddings[:,:i,:]
                # this makes the sequence look like ((s_1, a_1), r_1, Q_1, (s_2, a_2), ...)
                inputs = torch.cat((inputs, state_action_embeddings[:,i,:].unsqueeze(1)), dim=1) # (3 * sequence_length - 2, hidden_size)
                # feed in the input embeddings (not word indices as in NLP) to the model
                
                # fit the n_position size
                if inputs.size()[1] > self.n_positions:
                    inputs = inputs[:,-self.n_positions:,:]

                transformer_outputs = self.transformer(inputs_embeds=inputs)
                x = transformer_outputs['last_hidden_state'] # (3 * sequence_length - 2, hidden_size)

                # get predictions
                q_value_pred = self.predict_q_value(x) 
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
            for j in candidate_points:
                temp_state_actions = torch.from_numpy(np.array(next_state_actions))[:,i:i+1,j,:].float()
                state_action_embeddings_new = self.embed_state_action(temp_state_actions.to(self.device)) + time_embeddings[:,i:i+1,:]
                if i == 0:
                    state_action_embeddings_new = torch.cat((state_action_embeddings[:,:i,:], state_action_embeddings_new), dim = 1)
                    transformer_outputs = self.transformer(inputs_embeds=state_action_embeddings_new)
                    x = transformer_outputs['last_hidden_state']
                    temp.append(self.predict_q_value(x).detach())
                else:
                    
                    inputs = state_action_embeddings[:,:i,:]
                    # this makes the sequence look like ((s_1, a_1), r_1, Q_1, (s_2, a_2), ...)
                    inputs = torch.cat((inputs, state_action_embeddings_new), dim=1) # (3 * sequence_length - 2, hidden_size)
                    # feed in the input embeddings (not word indices as in NLP) to the model

                    # fit the n_position size
                    if inputs.size()[1] > self.n_positions:
                        inputs = inputs[:,-self.n_positions:,:]

                    transformer_outputs = self.transformer(inputs_embeds=inputs)
                    x = transformer_outputs['last_hidden_state'] # (3 * sequence_length - 2, hidden_size)

                    # get predictions
                    q_value_pred = self.predict_q_value(x) 
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
        
        inputs = state_action_embeddings[:,:pred_len-1,:]
        # this makes the sequence look like ((s_1, a_1), r_1, Q_1, (s_2, a_2), ...)
        inputs = torch.cat((inputs, state_action_embeddings[:,-1:,:]), dim=1) # (3 * sequence_length - 2, hidden_size)
        # feed in the input embeddings (not word indices as in NLP) to the model
        
        # fit the n_position size
        if inputs.size()[1] > self.n_positions:
            inputs = inputs[:,-self.n_positions:,:]

        transformer_outputs = self.transformer(inputs_embeds=inputs)
        x = transformer_outputs['last_hidden_state'] # (3 * sequence_length - 2, hidden_size)

        # get predictions
        q_value_pred = self.predict_q_value(x) 

        return q_value_pred
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.trunc_normal_(module.weight, mean=1.0/(1.0-self.gamma), std=0.02, a=0, b=1.0/(1.0-self.gamma)+0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.trunc_normal_(module.weight, mean=1.0/(1.0-self.gamma), std=0.02, a=0, b=1.0/(1.0-self.gamma)+0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    

    def select_action(self, state_actions, target_network):
        torch.cuda.empty_cache()
        # first action is random
        if len(self.trajectory) < self.initial_sample:
            action = random.randint(0, self.domain_size - 1)
            return action

        # epsilon greedy
        e = random.random()
        if e < self.epsilon:
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
        # action = np.argmax(q_values[:,-1,:].cpu())
        return action

    def update(self, target_network): 
        loss_value = 0
        self.optimizer.zero_grad()

        batch_size = min(self.batch_size, self.trajectory_buffer.tree.n_entries)
        batch, idxs, is_weights = self.trajectory_buffer.sample(batch_size)
        
        # get data from the batch
        state_actions = []
        rewards = []
        next_state_actions = [] 
        for traj in batch:
            state_action = []
            reward = []
            next_state_action = []
            for transit in traj:
                state_action.append(transit['state_action'])
                reward.append(np.array([transit['reward']]))
                next_state_action.append(transit['next_state_actions'])
            state_actions.append(np.array(state_action))
            rewards.append(np.array(reward))
            next_state_actions.append(next_state_action)

        state_actions = torch.from_numpy(np.array(state_actions)).float()
        rewards = torch.from_numpy(np.array(rewards)).float()
        # Max over actions
        # next_q_values = [torch.sum(rewards[:,i:,:]) for i in range(1,rewards.size(1))]
        next_q_values = target_network.forward_without_q(state_actions, rewards, next_state_actions).detach()
        # next_q_values = target_network.forward_without_q(state_actions, rewards, None).detach()
        
        # assign the final q value to be zero 
        next_q_values[:,-1,:] = 0
        next_q_values[:,-2,:] = rewards[:,-1,:]
        
        state_actions = state_actions.to(self.device)
        # (batch_size, sequence_length, state_action_dim)
        rewards = rewards.to(self.device)
        # (batch_size, sequence_length, 1)
        q_values = self.forward(state_actions, rewards[:,:-1,:], next_q_values[:,:-1,:])

        pred_len = q_values.size(1)
        loss = (rewards[:,-pred_len:,:] + self.gamma * next_q_values[:,-pred_len:,:] - q_values) ** 2 - q_values * self.bc_rate

        loss = torch.sum(loss, axis = 1) # sum over the sequence length dimension
        # update priority
        for i in range(batch_size):
            self.trajectory_buffer.update(idxs[i], loss[i].item())
        
        loss = torch.sum(torch.FloatTensor(is_weights).to(self.device) * loss.squeeze())
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        loss_value = loss.detach().cpu().item()
        
        return loss_value

    def reset_trajectory(self):
        self.trajectory = []
        if self.device != 'cpu': 
            torch.cuda.empty_cache()
            
    def append_sample(self, trajectory):
        state_actions = []
        rewards = []
        for transit in trajectory:
            state_actions.append(transit['state_action'])
            rewards.append(np.array([transit['reward']]))
            
        state_actions = torch.from_numpy(np.array(state_actions)).unsqueeze(0).float()
        rewards = torch.from_numpy(np.array(rewards)).unsqueeze(0).float()
        next_q_values = target_network.forward_without_q(state_actions, rewards, None).detach()
        next_q_values[:,-1,:] = 0
        state_actions = state_actions[:,:-1,:].to(self.device)
        rewards = rewards[:,:-1,:].to(self.device)
        next_q_values = next_q_values[:,1:,:] # ignore first q value
        q_values = self.forward(state_actions, rewards[:,:self.max_length-1,:], next_q_values[:,:self.max_length-1,:]).detach()
        if q_values.size()[1] < next_q_values.size()[1]:
            next_q_values = next_q_values[:,-q_values.size()[1]:,:]
            rewards = rewards[:,-q_values.size()[1]:,:]
            loss = torch.sum((rewards + self.gamma * next_q_values - q_values) ** 2)
        else:
            loss = torch.sum((rewards[:,1:,:] + self.gamma * next_q_values[:,1:,:] - q_values[:,1:,:]) ** 2)
        
        self.trajectory_buffer.add(loss.item(), trajectory)
        
if __name__ == '__main__':
    if not os.path.exists(f"records/"):
        os.mkdir(f"records")
    if not os.path.exists(f"models/"):
        os.mkdir(f"models")
    parser = argparse.ArgumentParser(description='Multi-objective BO')
    parser.add_argument('--device', type=str, default="0", help='gpu device')
    parser.add_argument('--learner', type=str, default='QT')
    parser.add_argument('--env', type=str, default='BO')
    parser.add_argument('--training_episode', type=int, default=2001)
    parser.add_argument('--T', type=int, default=100)
    parser.add_argument('--domain_size', type=int, default=1000)
    parser.add_argument('--f_num', type=int, default=2)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--n_batch', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=0.00001)
    parser.add_argument('--update_freq', type=int, default=1)
    parser.add_argument('--target_update_freq', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_layer', type=int, default=4)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--n_positions', type=int, default=11)
    parser.add_argument('--warmup_steps', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--function_type', type=str, default='train')
    parser.add_argument('--model_episode', type=int, default=0)
    parser.add_argument('--optimizer', type=str, default="adam")
    parser.add_argument('--dropout', type=float, default = 0.1)
    parser.add_argument('--epsilon', type=float, default = 0.1)
    parser.add_argument('--sample_rate', type=float, default=0.01)
    parser.add_argument('--demo_rate', type=float, default=0.01)
    parser.add_argument('--buffer_size', type=int, default=128)
    parser.add_argument('--new_reward',type=int, default=1)
    parser.add_argument('--ls_learned_freq', type=int, default=10, help='freq of learning ls')
    parser.add_argument('--perturb_noise_level', type=float, default=0.01, help='perturbed noise')
    parser.add_argument('--observation_noise_level', type=float, default=0.01, help='observation noise')
    parser.add_argument('--update_step', type=int, default=10, help='# of GD steps')
    parser.add_argument('--demo_time', type=int, default=0, help='time for demo')
    parser.add_argument('--demo_policy', type=str, default="qNEHVI", help='demo policy')
    parser.add_argument('--initial_sample', type=int, default=1, help='# of initial sample')
    parser.add_argument('--temperature', type=int, default=1000, help='temperature of softmax policy')
    parser.add_argument('--domain_dim', type=int, default=1, help='domain dimension')
    parser.add_argument('--online_ls', type=int, default=1, help='online learn ls')
    args = parser.parse_args()
    # f = open("record.txt", "a")
    # f.write(str(record_index) + "\t" + str(args) + "\n")
    # f.close()
    MOBO_info = str(args)
    print(MOBO_info)
    writer = SummaryWriter(f"logs/QT_fnum_{args.f_num}")
    
    if args.device != "cpu":
        device = torch.device("cuda:" + args.device if torch.cuda.is_available() else "cpu")
    else: 
        device = "cpu"

    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)

    # set seed for reproduc
    if args.seed > 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    
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
                 sample_rate = args.sample_rate,
                 device = device).to(device)
     
    # if args.model_episode != 0:
    #     learner.load_state_dict(torch.load("./model/episode_{}_{}.pth".format(args.model_episode, MOBO_info)))
    learner.train()

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
                 sample_rate = args.sample_rate,
                 device = device).to(device)
    
    target_network.load_state_dict(learner.state_dict())
    target_network.eval()

    
    env = Environment.env.Environment(T = args.T, 
                      domain_size = args.domain_size, 
                      f_num = args.f_num, 
                      function_type = args.function_type, 
                      seed = args.seed,
                      new_reward = args.new_reward,
                      perturb_noise_level = args.perturb_noise_level,
                      observation_noise_level = args.observation_noise_level,
                      domain_dim = args.domain_dim,
                      online_ls=args.online_ls)
    ehi = ExpectedHypervolumeImprovement(domain_size=env.domain_size, f_num=env.f_num, min_function_values=env.min_function_values)
    iterations = 0
    for e in range(args.training_episode):
        
        if np.random.binomial(1, args.demo_rate, 1) or e < args.demo_time:
            demo = 1
        else:
            demo = 0
    
        # initialization
        seed=args.seed+e*10
        env.reset(seed=seed)
        env.history['info'] = str(args)
        env.history['demo'] = demo
        losses = [] # training record
        lss = [] # training record
        sas = [] # training record
        
        # initial sample
        X = Environment.function_preprocessing.domain(args.function_type, args.domain_size, seed, args.domain_dim) 
        action = random.randint(0,args.domain_size-1)
        y_star, reward, regret = env.step(X[action])
        gp = Environment.env.GaussianProcess(np.array(env.history["x"]), np.array(env.history["y_observed"]), env.kernel, env.kernel_ls, env.f_num, dim = args.domain_dim)
        if args.online_ls:
            gp.fit(x = np.array(env.history["x"]), y = np.array(env.history["y_observed"]))
        state_actions = gp.construct_state_action_pair(X, y_star, 0/args.T)
        
        # state_actions = Environment.env.construct_state_action_pair(X, gp, y_star, 0)

        # record transition
        learner.trajectory.append({"state_action": state_actions[action], "reward": float(reward), "next_state_actions": state_actions})
            
        # training record
        lss.append(torch.cat([gp.GP[i].covar_module.lengthscale.detach().cpu() for i in range(args.f_num)], dim=1).numpy())
        sas.append(state_actions)

        if demo:
            gp_demo, _ = env.fit_gp(0)
            pred = gp_demo.posterior(torch.tensor(X, dtype=torch.double)).mean # gp(torch.tensor(X)).mean.T
            partitioning = FastNondominatedPartitioning(
                ref_point=torch.tensor(env.min_function_values, dtype=torch.double),
                Y=pred)
            
            standard_bounds = torch.zeros(2, env.domain_dim)
            standard_bounds[1] = 1
            
            if args.demo_policy == "qEHVI":
                demo_policy = qExpectedHypervolumeImprovement(
					model=gp_demo,
					ref_point=torch.tensor(env.min_function_values, dtype=torch.double),
					partitioning=partitioning,
				)
            elif args.demo_policy == "qNEHVI":
                demo_policy = qNoisyExpectedHypervolumeImprovement(
					model=gp_demo,
					ref_point=torch.tensor(env.min_function_values, dtype=torch.double),
					X_baseline=torch.tensor(env.history['x'], dtype=torch.double),
					prune_baseline=True,
				)
            elif args.demo_policy == "EHI":
                demo_policy = ExpectedHypervolumeImprovement(domain_size=env.domain_size, f_num=env.f_num, min_function_values=env.min_function_values)
            

        # training iterations
        for t in tqdm.tqdm(range(1, args.T)):
            
            # select action
            learner.eval()
            if demo:
                gp_demo, _ = env.fit_gp(t)
                pred = gp_demo.posterior(torch.tensor(X, dtype=torch.double)).mean # gp(torch.tensor(X)).mean.T
                partitioning = FastNondominatedPartitioning(
                    ref_point=torch.tensor(env.min_function_values, dtype=torch.double),
                    Y=pred)
                
                standard_bounds = torch.zeros(2, env.domain_dim)
                standard_bounds[1] = 1
                
                if args.demo_policy == "qEHVI":
                    demo_policy = qExpectedHypervolumeImprovement(
                        model=gp_demo,
                        ref_point=torch.tensor(env.min_function_values, dtype=torch.double),
                        partitioning=partitioning,
                    )
                elif args.demo_policy == "qNEHVI":
                    demo_policy = qNoisyExpectedHypervolumeImprovement(
                        model=gp_demo,
                        ref_point=torch.tensor(env.min_function_values, dtype=torch.double),
                        X_baseline=torch.tensor(env.history['x'], dtype=torch.double),
                        prune_baseline=True,
                    )
                elif args.demo_policy == "EHI":
                    demo_policy = ExpectedHypervolumeImprovement(domain_size=env.domain_size, f_num=env.f_num, min_function_values=env.min_function_values)
                
                candidates, _ = optimize_acqf(
                    acq_function=demo_policy,
                    bounds=standard_bounds.double(),
                    q=1,
                    num_restarts=1,
                    raw_samples=1,
                    options={"batch_limit": 1, "maxiter": 10},
                    sequential=True,
                )
                x_best = candidates.detach().numpy()[0]
            else:
                action = learner.select_action(state_actions, target_network)
                x_best = X[action]
            learner.train()

            # env update
            y_star, reward, regret = env.step(x_best)
            
            X = Environment.function_preprocessing.domain(args.function_type, args.domain_size, seed+t, d = args.domain_dim)
            gp = Environment.env.GaussianProcess(np.array(env.history["x"]), np.array(env.history["y_observed"]), env.kernel, env.kernel_ls, env.f_num, dim = args.domain_dim)
            # learn ls for GP
            if args.online_ls and t % args.ls_learned_freq == 0:
                env.kernel_ls = gp.fit(x = np.array(env.history["x"]), y = np.array(env.history["y_observed"]))
                
            next_state_actions = gp.construct_state_action_pair(X, y_star, t/args.T) 
            # next_state_actions = Environment.env.construct_state_action_pair(X, gp, y_star, t/args.T)

            # record transition
            learner.trajectory.append({"state_action": state_actions[action], "reward": float(reward), "next_state_actions": next_state_actions})
            
            # update current state
            state_actions = next_state_actions

            # training record
            lss.append(torch.cat([gp.GP[i].covar_module.lengthscale.detach().cpu() for i in range(args.f_num)], dim=1).numpy())
            sas.append(state_actions)

        # record final transition
        # learner.trajectory.append({"state_action": state_actions[action], "reward": 0.0, "next_state_actions": next_state_actions})

        # learner update
        learner.append_sample(learner.trajectory)
        learner.reset_trajectory()
        i = 0
        while i < args.update_step:
            i+=1
            loss = learner.update(target_network)
            losses.append(loss)
            writer.add_scalar("train/loss", loss, iterations)
            iterations += 1
            if loss <= 0.0001:
                break
        env.history["loss"] = losses
        writer.add_scalar("train/final_regret", regret, e)
        writer.add_scalar("train/mean_loss", np.mean(losses), e)
        # update target network
        if e % args.target_update_freq == 0:
            target_network.load_state_dict(learner.state_dict())
            target_network.eval()

        if demo:
            print('EP:{} | Loss: {:.3f} | Q: None | R: {:.3f}'.format(e + args.model_episode, np.mean(losses), regret))
        else:
            print('EP:{} | Loss: {:.3f} | Q: {:.3f} | R: {:.3f}'.format(e + args.model_episode, np.mean(losses), learner.previous_q_values[:,-1,:].item(), regret))

        # save model
        if e % 5 == 0: torch.save(learner.state_dict(), "./models/QT_fnum_{}_{}.pth".format(args.f_num, e + args.model_episode))

        # save regrets
        filename = "QT_fnum_{}_{}.pth".format(args.f_num, e + args.model_episode)
        with open(os.path.join(f"records", filename), 'wb') as f:
            pickle.dump(env.history, f)
        
    torch.save(learner.state_dict(), "./models/QT_fnum_{}_{}.pth".format(args.f_num, e + args.model_episode))