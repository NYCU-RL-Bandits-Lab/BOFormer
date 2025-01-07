from transformers import T5Tokenizer, T5Model
import os
import Environment.env
import Environment.function_preprocessing
import argparse
import numpy as np
import torch
import random
import pickle
import tqdm
import pathlib
from torch.utils.tensorboard import SummaryWriter
from ehi import ExpectedHypervolumeImprovement
import sentencepiece

def format_array(arr, name):
    ordinal_suffix = {1: "st", 2: "nd", 3: "rd"}
    formatted_values = [f"{i}{ordinal_suffix.get(i, 'th')}_" + name + "=" + str(round(val, 3)) for i, val in enumerate(arr, start=1)]
    return " ".join(formatted_values)

class optformer(torch.nn.Module):
	def __init__(self, lr = 0.01, weight_decay = 0.01, f_num = 1, max_length = 512) -> None:
		super(optformer, self).__init__()
		self.tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-base")
		self.model = T5Model.from_pretrained("google-t5/t5-base")
		self.max_length = max_length
		self.optimizer = torch.optim.AdamW(params=self.parameters(), lr=lr, weight_decay=weight_decay)
		self.f_num = f_num
		# the forward function automatically creates the correct decoder_input_ids
		def is_number(string):
			try:
				float_value = float(string)
				int_value = int(string)
				if float_value >= 0 and int_value >=0 and float_value <= 1000 and int_value <= 1000:
					return True
			except ValueError:
				return False
		self.vocab_list = list(self.tokenizer.get_vocab().keys())
		self.number_index = []
		for i in range(len(self.vocab_list)):
			if is_number(self.vocab_list[i]):
				self.number_index.append(i)
		self.number_list = [int(self.vocab_list[i]) for i in self.number_index]

		# self.number_embed = torch.nn.Linear(768, len(self.number_index))
		self.logits_embed = torch.nn.Linear(768, len(self.vocab_list))
		self.ordinal_suffix = {1: "st", 2: "nd", 3: "rd"}

	def select_action(self, meta_data, trails, domain, y_star):
		eis = []
		encoder_input = self.tokenizer(meta_data, return_tensors="pt").input_ids
		domain_candidate = random.sample(domain.tolist(), int(len(domain)/20))
		for x in domain_candidate:
			x = trails + format_array(x, "dimension_x")
			x = x[len(x)-self.max_length:]
			ei=1
			for f in range(self.f_num):
				decoder_input = self.tokenizer(x, return_tensors="pt").input_ids
				decoder_outputs = self.model(input_ids = encoder_input.cuda(), decoder_input_ids = decoder_input.cuda())['last_hidden_state'][0,-1,:]
				vocab_logits = self.logits_embed(decoder_outputs).detach()
				number_prob = torch.nn.functional.softmax(vocab_logits[self.number_index], dim = 0).cpu()
				y_pred = np.sum([max(self.number_list[i] - y_star[f],0) * number_prob[i]  for i in range(len(self.number_list))])
				# torch.sum(torch.multiply(torch.tensor(self.number_list), number_prob.cpu()))
				ei *= y_pred
				x += " {}{}_objective_y={:.3f}".format(f+1, self.ordinal_suffix.get(f+1, 'th'),y_pred)
			eis.append(ei)
		x = domain_candidate[np.argmax(eis)]
		return np.where(domain==x)[0][0]
	
	def train(self, encoder_input, decoder_input):
		self.optimizer.zero_grad()
		loss = 0
		for i in range(decoder_input.size()[1]-1): # num of token
			decoder_outputs = self.model(input_ids = encoder_input.cuda(), decoder_input_ids = decoder_input[:,:i+1].cuda())['last_hidden_state'][0,-1,:]
			vocab_logits = self.logits_embed(decoder_outputs)
			prob = torch.nn.functional.softmax(vocab_logits, dim = 0)
			loss += -torch.log(prob[decoder_input[0,i+1]])
		loss.backward()
		self.optimizer.step()
		return loss.item()

if __name__== '__main__':
	os.makedirs(f"results/optformer", exist_ok=True)

	parser = argparse.ArgumentParser(description='Multi-objective BO')
	parser.add_argument('--device', type=int, default=0, help='gpu device')
	parser.add_argument('--seed', type=int, default=1, help='random seed of numpy, torch and random')
	parser.add_argument('--function_type', type=str, default="RBF_0.05", help='RBF_0.05, RBF_0.2, RBF_0.3, matern52_0.05, matern52_0.2, matern52_0.3, BC, AR, ARS, DRZ, Branin, Currin, YAHPO')
	parser.add_argument('--yahpo_scenario', type=str, default='lcbench', help='lcbench, rbv2_xgboost, rbv2_svm, rbv2_glmnet')
	parser.add_argument('--NERF_scene', type=str, default="lego", help='NERF_scene')
	parser.add_argument('--domain_size', type=int, default=1000, help='domain size')
	parser.add_argument('--f_num', type=int, default=2, help='number of objective function')
	parser.add_argument('--T', type=int, default=100, help='total iteration')
	parser.add_argument('--episode', type=int, default=100, help='number of episodes')
	parser.add_argument('--max_length', type=int, default=128, help='string length')
	parser.add_argument('--ls_learned_freq', type=int, default=10, help='freq of learning ls')
	parser.add_argument('--perturb_noise_level', type=float, default=0.1, help='perturbed noise')
	parser.add_argument('--observation_noise_level', type=float, default=0.1, help='observation noise')
	
	args = parser.parse_args()
	
	torch.cuda.set_device(args.device)
	if args.seed > 0:
		np.random.seed(args.seed)
		torch.manual_seed(args.seed)
		random.seed(args.seed)
  
	learner = optformer(f_num = args.f_num, max_length = args.max_length).cuda()
	learner.load_state_dict(torch.load("./optformer_models/optformer_train_f_num_{}_demo_1_990.pth".format(args.f_num)))
	# learner.eval()
	# trails_list = []
	# meta_data_list = []
	hvs_list = []
	env = Environment.env.Environment(T = args.T, 
                      domain_size = args.domain_size, 
                      f_num = args.f_num, 
                      function_type = args.function_type, 
                      yahpo_scenario= args.yahpo_scenario, 
					  NERF_scene = args.NERF_scene,
                      seed = args.seed,
                      new_reward = 0,
					  online_ls = 1,
                      perturb_noise_level = args.perturb_noise_level,
                      observation_noise_level = args.observation_noise_level,
                      ls_learned_freq = args.ls_learned_freq)
	
	for e in range(args.episode):
		seed = args.seed + e*10
		env.reset(seed=seed, episode=e)
		env.history['info'] = str(args)

		args.domain_size = np.shape(env.X)[0]
		# initial sample
		X = Environment.function_preprocessing.domain(args.function_type, args.domain_size, seed) 
		y_star, reward, regret = env.step(X[random.randint(0,args.domain_size-1)])
		gp, _ = env.fit_gp(0)
		ls = env.ls
		trails = ""
		
		for t in tqdm.tqdm(range(1, args.T)):
			
			meta_data = "length_scale=" + "[" + ', '.join(str(round(x.mean().item(),3)) for x in ls) + "]" # + \
					# " kernel=" + "[" + ', '.join(str(x) for x in kernel) + "]"
			
			action = learner.select_action(meta_data, trails, X, y_star = y_star)
			# observe y value
			y_star, reward, regret = env.step(X[action])
			trails += format_array(env.history['x'][-1], "dimension_x") + " " + format_array(env.history['y_observed'][-1], "objective_y") + " "
			
			# learn ls for GP
			gp, _ = env.fit_gp(t)
			ls = env.ls

			X = Environment.function_preprocessing.domain(args.function_type, args.domain_size, seed+t) 

		# save history
		if args.function_type == "YAHPO":
			with open("./results/optformer/{}_{}_{}_f_num_{}_per_{}_obs_{}_{}.pkl".format("optformer", args.function_type, args.yahpo_scenario, args.f_num, e), 'wb') as f:
				pickle.dump(env.history, f)
		elif args.function_type.startswith("NERF"):
			with open("./results/optformer/{}_{}_{}_f_num_{}_per_{}_obs_{}_{}.pkl".format("optformer", args.function_type, args.NERF_scene, args.f_num, args.perturb_noise_level, args.observation_noise_level, e), 'wb') as f:
				pickle.dump(env.history, f)
		else:
			with open("./results/optformer/{}_{}_f_num_{}_{}.pkl".format("optformer", args.function_type, args.f_num, e), 'wb') as f:
				pickle.dump(env.history, f)
