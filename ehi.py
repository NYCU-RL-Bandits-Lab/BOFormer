import numpy as np
from pymoo.factory import get_performance_indicator
from pymoo.config import Config

class ExpectedHypervolumeImprovement:
	def __init__(self, domain_size, f_num, min_function_values):
		self.domain_size = domain_size
		self.f_num = f_num
		self.min_function_values = min_function_values


	def countHypervolume(self, function_values, min_function_values):
		hv = get_performance_indicator("hv", ref_point=-1*min_function_values)
		return float(hv.do(-1*function_values))


	def select_action(self, state_action_pairs):
		mu = state_action_pairs[:, :self.f_num]
		sigma = state_action_pairs[:, self.f_num:self.f_num*2]

		# number sample from each Gaussian 
		sample_Gaussian_num = 2
		ExI = []

		for i in range(self.domain_size):
			y = []
			for j in range(self.f_num):
				sample_Gaussian = np.random.normal(mu[i][j], sigma[i][j], sample_Gaussian_num)
				y.append(sample_Gaussian.reshape(-1, 1))
			y = np.concatenate(y, axis=1)	# len(y) = sample_Gaussian_num, len(y[i]) = f_num

			improvement = []
			# -yt + min_point
			for j in range(sample_Gaussian_num-1):
				improvement.append(self.countHypervolume(y[:j+2], self.min_function_values) - self.countHypervolume(y[:j+1], self.min_function_values))
			ExI.append(sum(improvement)/sample_Gaussian_num)

		action = np.argmax(ExI)
		
		return action