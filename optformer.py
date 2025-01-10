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
		domain_candidate = random.sample(domain.tolist(), int(len(domain)))
		for x in domain_candidate:
			x = trails + format_array(x, "dimension_x")
			x = x[len(x)-self.max_length:]
			ei=1
			for f in range(self.f_num):
				decoder_input = self.tokenizer(x, return_tensors="pt").input_ids
				decoder_outputs = self.model(input_ids = encoder_input.cuda(), decoder_input_ids = decoder_input.cuda())['last_hidden_state'][0,-1,:]
				vocab_logits = self.logits_embed(decoder_outputs).detach()
				number_prob = torch.nn.functional.softmax(vocab_logits[self.number_index]).cpu()
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
	record_index = 0
	while pathlib.Path(f"optformer_logs/logs{record_index}").exists():
		record_index += 1
	writer = SummaryWriter(f"optformer_logs/logs{record_index}") # tensorboard --logdir=logs{i}
	os.makedirs(f"optformer_models/", exist_ok=True)
 
	parser = argparse.ArgumentParser(description='Multi-objective BO')
	parser.add_argument('--device', type=int, default=0, help='gpu device')
	parser.add_argument('--seed', type=int, default=1, help='random seed of numpy, torch and random')
	parser.add_argument('--function_type', type=str, default="train", help='RBF_0.05, RBF_0.2, RBF_0.3, matern52_0.05, matern52_0.2, matern52_0.3, BC, AR, ARS, DRZ, Branin, Currin')
	parser.add_argument('--domain_size', type=int, default=1000, help='domain size')
	parser.add_argument('--f_num', type=int, default=2, help='number of objective function')
	parser.add_argument('--T', type=int, default=100, help='total iteration')
	parser.add_argument('--episode', type=int, default=1000, help='number of episodes')
	parser.add_argument('--max_length', type=int, default=128, help='string length')
	parser.add_argument('--demo', type=int, default=1, help='demo policy')
	parser.add_argument('--ls_learned_freq', type=int, default=10, help='freq of learning ls')
	parser.add_argument('--perturb_noise_level', type=float, default=0.1, help='perturbed noise')
	parser.add_argument('--observation_noise_level', type=float, default=0.1, help='observation noise')
	parser.add_argument('--update_step', type=int, default=10, help='# of GD steps')
	args = parser.parse_args()
	
	torch.cuda.set_device(args.device)
	if args.seed > 0:
		np.random.seed(args.seed)
		torch.manual_seed(args.seed)
		random.seed(args.seed)
  
	learner = optformer(f_num = args.f_num, max_length = args.max_length).cuda()
	# trails_list = []
	# meta_data_list = []
	hvs_list = []
	env = Environment.env.Environment(T = args.T, 
							domain_size = args.domain_size, 
							f_num = args.f_num, 
							function_type = args.function_type, 
							seed = args.seed,
							new_reward = False,
							perturb_noise_level = args.perturb_noise_level,
                      		observation_noise_level = args.observation_noise_level,)
	iterations = 0
	for e in range(args.episode):
		seed = args.seed + e*10
		env.reset(seed=seed)
		env.history['info'] = str(args)

		# initial sample
		X = Environment.function_preprocessing.domain(args.function_type, args.domain_size, seed) 
		y_star, reward, regret = env.step(X[random.randint(0,args.domain_size-1)])
		gp = env.fit_gp()
		ls = torch.flatten(gp.covar_module.base_kernel.lengthscale.detach().cpu()).numpy()
		state_actions = Environment.env.construct_state_action_pair(X, gp, y_star, 0)

		ehi = ExpectedHypervolumeImprovement(domain_size=env.domain_size, f_num=env.f_num, min_function_values=env.min_function_values)
		trails = ""
		
		for t in tqdm.tqdm(range(1, args.T)):
			
			meta_data = "length_scale=" + "[" + ', '.join(str(round(x,3)) for x in ls) + "]" # + \
					# " kernel=" + "[" + ', '.join(str(x) for x in kernel) + "]"

			# select action
			if args.demo:
				action = ehi.select_action(state_actions)
			else:
				action = learner.select_action(meta_data, trails, X, y_star = y_star)
				# observe y value
			y_star, reward, regret = env.step(X[action])
			trails += format_array(env.history['x'][-1], "dimension_x") + " " + format_array(env.history['y_observed'][-1], "objective_y") + " "
			
			# learn ls for GP
			if t % args.ls_learned_freq == 0:
				gp = env.fit_gp()
				ls = torch.flatten(gp.covar_module.base_kernel.lengthscale.detach().cpu()).numpy()

			X = Environment.function_preprocessing.domain(args.function_type, args.domain_size, seed+t) 
			state_actions = Environment.env.construct_state_action_pair(X, gp, y_star, t/args.T)

		i = 0
		losses = []
		while i < args.update_step:
			i+=1
			trails = trails[len(trails)-args.max_length:]
			encoder_input = learner.tokenizer(meta_data, return_tensors="pt").input_ids.detach()
			decoder_input = learner.tokenizer(trails, return_tensors="pt").input_ids.detach()
			loss = learner.train(encoder_input, decoder_input)
			losses.append(loss)
			writer.add_scalar("train/loss", loss, iterations)
			iterations += 1
			if loss <= 0.1:
				break
		env.history["loss"] = losses
		writer.add_scalar("train/mean_loss", np.mean(losses), e)

		# save history
		with open("./optformer_models/{}_{}_f_num_{}_demo_{}_{}.pkl".format("optformer", args.function_type, args.f_num, args.demo, e), 'wb') as f:
			pickle.dump(env.history, f)
		# save model
		if e % 10 == 0:
			torch.save(learner.state_dict(), "./optformer_models/{}_{}_f_num_{}_demo_{}_{}.pth".format("optformer", args.function_type, args.f_num, args.demo, e))
  
