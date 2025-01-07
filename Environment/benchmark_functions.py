import math
import numpy as np
import random
from sklearn.gaussian_process.kernels import RBF, Matern
from scipy.interpolate import interp1d, LinearNDInterpolator, NearestNDInterpolator
import pickle
import sobol_seq
import pandas
class LinearNDInterpolatorExt(object):
	def __init__(self, points,values):
		self.funcinterp=LinearNDInterpolator(points,values)
		self.funcnearest=NearestNDInterpolator(points,values)
	def __call__(self,*args):
		t=self.funcinterp(*args)
		if not np.isnan(t):
			return t.item(0)
		else:
			return self.funcnearest(*args)

def set_noise_level(x):
	global noise_level
	noise_level = x

def domain(function_type, domain_num, seed, d = 1):
	if (function_type == "train_05" or function_type == "train_005" or function_type == "train" or function_type == "train_large" or function_type == "RBF_0.05" or function_type == "RBF_0.2"  or function_type == "RBF_0.3" or function_type == "matern52_0.05" or function_type == "matern52_0.2" or function_type == "matern52_0.3"):
		X = sobol_seq.i4_sobol_generate(d, domain_num, seed)
	elif (function_type == "RE24" or function_type == "DR" or function_type == "BC" or function_type == "ARS" or function_type == "DRZ" or function_type == "Branin" or function_type == "Currin" or function_type == "AR" or function_type == "DR" or function_type == "ARa" or function_type == "BCD" or function_type == "ASR"):
		X = sobol_seq.i4_sobol_generate(2, domain_num, seed)
	elif (function_type == "RE25" or function_type == "RE22" or function_type == "RE31"):
		X = sobol_seq.i4_sobol_generate(3, domain_num, seed)
	elif (function_type == "RE33" or function_type == "RE21" or function_type == "RE23" or function_type == "RE32" or function_type == "RE36" or function_type == "RE37" or function_type == "YAHPO"):
		X = sobol_seq.i4_sobol_generate(4, domain_num, seed)
	elif (function_type == "RE34" or function_type == "NERF"):
		X = sobol_seq.i4_sobol_generate(5, domain_num, seed)
	elif (function_type == "RE35"):
		X = sobol_seq.i4_sobol_generate(7, domain_num, seed)
	return X

def norm_add_noise(y, max_f, min_f, disable = False):
	if disable:
		return y
	global noise_level
	if y == np.inf or y > max_f:
		return (max_f * (1 + random.uniform(-noise_level, noise_level)) - min_f) / (max_f - min_f)
	elif np.isnan(y) or y < min_f:
		return (min_f * (1 + random.uniform(-noise_level, noise_level)) - min_f) / (max_f - min_f)
	else:
		return (y * (1 + random.uniform(-noise_level, noise_level)) - min_f) / (max_f - min_f)
	
def norm(y, max_f, min_f):
	global noise_level
	if y == np.inf or y > max_f:
		return (max_f - min_f) / (max_f - min_f)
	elif np.isnan(y) or y < min_f:
		return (min_f - min_f) / (max_f - min_f)
	else:
		return (y - min_f) / (max_f - min_f)

HPO_index = -1

def get_HPO_index():
	global HPO_index
	return HPO_index

def set_HPO_index(x):
	global HPO_index
	HPO_index = x

def HPO(x):
	global HPO_index
	file = open("../Environment/HPObench/hpobenchXGB_" + str(HPO_index) + ".pkl",'rb')
	data = pickle.load(file)
	max_f = max(data['accs'])
	min_f = min(data['accs'])
	if list(x) in data['domain']:
		obj = data['accs'][data['domain'].index(list(x))]
	else:
		obj = np.argmin(np.sum(abs(np.array(x)-np.array(data['domain'])), axis=1), axis=0)
	return norm_add_noise(obj, max_f, min_f)

def GPF(X, kernel, kernel_ls):
	def countKernel(x, y, kernel, kernel_ls):  # x and y must be a 2-dimensional array
		kernel_var = 1.0
		if (kernel == "RBF"):
			kernelF = kernel_var * RBF(kernel_ls)
		elif (kernel == "matern52"):
			kernelF = kernel_var * Matern(length_scale=kernel_ls, nu=2.5)
		cov = kernelF.__call__(x, y)
		return cov
	mu = np.zeros(len(X))       # mu = 0
	cov = countKernel(X, X, kernel, kernel_ls)
	Y = np.random.multivariate_normal(mu, cov).reshape(len(X), 1)
	Y = (Y - min(Y)) / (max(Y) - min(Y)) # normalize
	if np.shape(X)[1] != 1:
		f = f = LinearNDInterpolatorExt(X, Y.reshape(-1))
	else:
		f = interp1d(X.reshape(-1), Y.reshape(-1), kind='cubic', fill_value='extrapolate')  # f(x) = y
	return f

def SE(X, kernel_ls):
	mu = np.zeros(len(X))       # mu = 0
	cov = RBF(kernel_ls).__call__(X, X)     # x and y must be a 2-dimensional array
	Y = np.random.multivariate_normal(mu, cov).reshape(len(X), 1)
	Y = (Y - min(Y)) / (max(Y) - min(Y)) # normalize
	f = interp1d(X.reshape(-1), Y.reshape(-1), kind='cubic', fill_value='extrapolate')  # f(x) = y
	return f

def Matern52(X, kernel_ls):
	mu = np.zeros(len(X))       # mu = 0
	cov = Matern(length_scale=kernel_ls, nu=2.5).__call__(X, X)     # x and y must be a 2-dimensional array
	Y = np.random.multivariate_normal(mu, cov).reshape(len(X), 1)
	Y = (Y - min(Y)) / (max(Y) - min(Y)) # normalize
	f = interp1d(X.reshape(-1), Y.reshape(-1), kind='cubic', fill_value='extrapolate')  # f(x) = y
	return f

def Branin(x):
	# scale x from [0, 1] to truth domain
	xmin = 0
	xmax = 10
	x = list(xmin + np.asarray(x) * (xmax - xmin))

	# f
	b = 5.1 / (4 * pow(math.pi, 2))
	c = 5. / math.pi
	r = 6
	s = 10
	t = 1. / (8 * math.pi)

	y = float(pow((x[1] - b * pow(x[0], 2) + c * x[0]- r), 2) + s * (1-t) * np.cos(x[0]) + s)
	max_f = 100.
	min_f = 0.
	return norm_add_noise(y, max_f, min_f)

def Currin(x):
	# scale x from [0, 1] to truth domain
	xmin = 0
	xmax = 1
	x = list(xmin + np.asarray(x) * (xmax - xmin))

	# f
	y = float(((1 - math.exp(-0.5*(1/x[1]))) * ((2300*pow(x[0], 3) + 1900*pow(x[0], 2) + 2092*x[0] + 60) / (100*pow(x[0], 3) + 500*pow(x[0], 2) + 4*x[0] + 20))))
	max_f = 14.
	min_f = 0.

	return norm_add_noise(y, max_f, min_f)

def Ackley(x):
	# scale x from [0, 1] to truth domain
	xmin = -30
	xmax = 30
	x = list(xmin + np.asarray(x) * (xmax - xmin))

	# f
	square_sum = pow(x[0], 2) + pow(x[1], 2)
	cos_sum = math.cos(2 * math.pi * x[0]) + math.cos(2 * math.pi * x[1])
	y = -20.0 * math.exp(-0.2 * math.sqrt(0.5*square_sum)) - math.exp(0.5 * cos_sum) + 20 + math.exp(1)

	max_f = 0.
	min_f = -28.

	return norm_add_noise(-y, max_f, min_f)

def Rosen(x):
	# scale x from [0, 1] to truth domain
	xmin = -2
	xmax = 2
	x = list(xmin + np.asarray(x) * (xmax - xmin))

	# f
	y = pow((x[0] - 1), 2) + 100 * pow((x[1] - pow(x[0], 2)), 2)

	max_f = 0.
	min_f = -4400.

	return norm_add_noise(-y, max_f, min_f)

def Sphere(x):
	# scale x from [0, 1] to truth domain
	xmin = -5
	xmax = 5
	x = list(xmin + np.asarray(x) * (xmax - xmin))

	# f
	y = pow(x[0], 2) + pow(x[1], 2)
	
	# perturb
	max_f = 0.
	min_f = -50.

	return norm_add_noise(-y, max_f, min_f)

def Schwefel(x):
	d = 2

	# scale x from [0, 1] to truth domain
	xmin = -500
	xmax = 500
	x = list(xmin + np.asarray(x) * (xmax - xmin))

	# f
	sum_ = 0
	for i in range(0, d):
		sum_ = sum_ + x[i] * math.sin(math.sqrt(abs(x[i])))
	y = 418.9829 * d - sum_
 
	max_f = 0
	min_f = -1676.

	return norm_add_noise(-y, max_f, min_f)

def Dixon(x):
	d = 2

	# scale x from [0, 1] to truth domain
	xmin = -10
	xmax = 10
	x = list(xmin + np.asarray(x) * (xmax - xmin))

	# f
	sum_ = 0
	for i in range(1, d):    
		sum_ = sum_ + (i+1) * pow((2 * pow(x[i], 2) - x[i-1]), 2)
	y = pow(x[0] - 1, 2) + sum_
	
	max_f = 0.
	min_f = -88311.

	return norm_add_noise(-y, max_f, min_f)

def Rastrigin(x):
	d = 2

	# scale x from [0, 1] to truth domain
	xmin = -5
	xmax = 5
	x = list(xmin + np.asarray(x) * (xmax - xmin))

	# f
	sum_ = 0
	for i in range(0, d):
		sum_ = sum_ + (pow(x[i], 2) - 10 * math.cos(2 * math.pi * x[i]))
	y = 10 * d + sum_

	max_f = 0.
	min_f = -81.

	return norm_add_noise(-y, max_f, min_f)

def Zakharov(x):
	d = 2

	# scale x from [0, 1] to truth domain
	xmin = -5
	xmax = 10
	x = list(xmin + np.asarray(x) * (xmax - xmin))

	# f
	sum1 = 0
	sum2 = 0
	for i in range(0, d):
		sum1 = sum1 + pow(x[i], 2)
		sum2 = sum2 + 0.5 * (i+1) * x[i]
	y = sum1 + pow(sum2, 2) + pow(sum2, 4)
	
	max_f = 0.
	min_f = -51041.

	return norm_add_noise(-y, max_f, min_f)

def optimization_function(function_type, function, domain_num, seed):
	if function_type != "HPO":
		try:
			X = np.load("./domain/domain_{}_{}.npy".format(function_type, domain_num))
		except:
			X = domain(function_type, domain_num, seed)
	else:
		global HPO_index
		file = open("./Environment/HPObench/hpobenchXGB_" + str(HPO_index) + ".pkl",'rb')
		data = pickle.load(file)
		X = data['domain']
	#X = domain(function_type, domain_num)

	F = function
	Y = []
	for j in X:
		Y.append(F(j))
	
	# domain minimum points
	domain_min_points = float(Y[np.argmin(Y)])

	# domain maximum points
	domain_max_points = float(Y[np.argmax(Y)])

	# log marginal likelihood
	# kernel_ls = lml(X, Y)

	#return F, Y, domain_max_points, domain_min_points, kernel_ls
	return F, Y, domain_max_points, domain_min_points

def GP_function(domain, function_type, dim = 1):

	# select ls
	if (function_type == "train"):
		kernel_ls = random.uniform(0.1, 0.4)
	elif (function_type == "train_large"):
		kernel_ls = random.uniform(0.4, 0.8)
	elif (function_type == "train_05"):
		kernel_ls = [0.5 for _ in range(dim)]
	elif (function_type == "train_005"):
		kernel_ls = [0.05 for _ in range(dim)]
	elif (function_type == "RBF_0.05" or function_type == "matern52_0.05"):
		kernel_ls = random.uniform(0.05, 0.1)
	elif (function_type == "RBF_0.2" or function_type == "matern52_0.2"):
		kernel_ls = random.uniform(0.2, 0.3)

	# select kernel
	if (function_type == "train" or function_type == "train_large" or function_type == "train_005" or function_type == "train_05"):
		GP_kernel = random.choices(["RBF", "matern52"])[0]
		F = GPF(domain, GP_kernel, kernel_ls)
	elif (function_type == "RBF_0.05" or function_type == "RBF_0.2" or function_type == "RBF_0.3"):
		GP_kernel = "RBF"
		F = SE(domain, kernel_ls)
	elif (function_type == "matern52_0.05" or function_type == "matern52_0.2" or function_type == "matern52_0.3"):
		GP_kernel = "matern52"
		F = Matern52(domain, kernel_ls)
	return F, GP_kernel, kernel_ls

def RE31_1(x):
	problem_name = 'RE31'
	n_objectives = 3
	n_variables = 3
	n_constraints = 0
	n_original_constraints = 3
	ubound = np.zeros(n_variables)
	lbound = np.zeros(n_variables)
	lbound[0] = 0.00001
	lbound[1] = 0.00001
	lbound[2] = 1.0
	ubound[0] = 100.0
	ubound[1] = 100.0
	ubound[2] = 3.0
	x = x*(ubound - lbound) + lbound

	f = np.zeros(n_objectives)
	g = np.zeros(n_original_constraints)

	x1 = x[0]
	x2 = x[1]
	x3 = x[2]

	# First original objective function
	f[0] = x1 * np.sqrt(16.0 + (x3 * x3)) + x2 * np.sqrt(1.0 + x3 * x3)
	# Second original objective function
	f[1] = (20.0 * np.sqrt(16.0 + (x3 * x3))) / (x1 * x3)

	# Constraint functions 
	g[0] = 0.1 - f[0]
	g[1] = 100000.0 - f[1]
	g[2] = 100000 - ((80.0 * np.sqrt(1.0 + x3 * x3)) / (x3 * x2))
	g = np.where(g < 0, -g, 0)                
	f[2] = g[0] + g[1] + g[2]
	
	obj = f[0]
	max_f = 813.7646219455767
	min_f = 0.23316847407085803
	return norm_add_noise(obj, max_f, min_f)

def RE31_2(x):
	problem_name = 'RE31'
	n_objectives = 3
	n_variables = 3
	n_constraints = 0
	n_original_constraints = 3
	ubound = np.zeros(n_variables)
	lbound = np.zeros(n_variables)
	lbound[0] = 0.00001
	lbound[1] = 0.00001
	lbound[2] = 1.0
	ubound[0] = 100.0
	ubound[1] = 100.0
	ubound[2] = 3.0
	x = x*(ubound - lbound) + lbound

	f = np.zeros(n_objectives)
	g = np.zeros(n_original_constraints)

	x1 = x[0]
	x2 = x[1]
	x3 = x[2]

	# First original objective function
	f[0] = x1 * np.sqrt(16.0 + (x3 * x3)) + x2 * np.sqrt(1.0 + x3 * x3)
	# Second original objective function
	f[1] = (20.0 * np.sqrt(16.0 + (x3 * x3))) / (x1 * x3)

	# Constraint functions 
	g[0] = 0.1 - f[0]
	g[1] = 100000.0 - f[1]
	g[2] = 100000 - ((80.0 * np.sqrt(1.0 + x3 * x3)) / (x3 * x2))
	g = np.where(g < 0, -g, 0)                
	f[2] = g[0] + g[1] + g[2]
	
	obj = f[1]
	max_f = 2223311.6192850918
	min_f = 0.3333343733738119
	return norm_add_noise(obj, max_f, min_f)

def RE31_3(x):
	problem_name = 'RE31'
	n_objectives = 3
	n_variables = 3
	n_constraints = 0
	n_original_constraints = 3
	ubound = np.zeros(n_variables)
	lbound = np.zeros(n_variables)
	lbound[0] = 0.00001
	lbound[1] = 0.00001
	lbound[2] = 1.0
	ubound[0] = 100.0
	ubound[1] = 100.0
	ubound[2] = 3.0
	x = x*(ubound - lbound) + lbound

	f = np.zeros(n_objectives)
	g = np.zeros(n_original_constraints)

	x1 = x[0]
	x2 = x[1]
	x3 = x[2]

	# First original objective function
	f[0] = x1 * np.sqrt(16.0 + (x3 * x3)) + x2 * np.sqrt(1.0 + x3 * x3)
	# Second original objective function
	f[1] = (20.0 * np.sqrt(16.0 + (x3 * x3))) / (x1 * x3)

	# Constraint functions 
	g[0] = 0.1 - f[0]
	g[1] = 100000.0 - f[1]
	g[2] = 100000 - ((80.0 * np.sqrt(1.0 + x3 * x3)) / (x3 * x2))
	g = np.where(g < 0, -g, 0)                
	f[2] = g[0] + g[1] + g[2]
	
	obj = f[2]
	max_f = 3765483.941211022
	min_f = 0.13316847407085802
	return norm_add_noise(obj, max_f, min_f)

def RE32_1(x):
	problem_name = 'RE32'
	n_objectives = 3
	n_variables = 4
	n_constraints = 0
	n_original_constraints = 4
	ubound = np.zeros(n_variables)
	lbound = np.zeros(n_variables)
	lbound[0] = 0.125
	lbound[1] = 0.1
	lbound[2] = 0.1
	lbound[3] = 0.125
	ubound[0] = 5.0
	ubound[1] = 10.0
	ubound[2] = 10.0
	ubound[3] = 5.0
	x = x*(ubound - lbound) + lbound
	f = np.zeros(n_objectives)
	g = np.zeros(n_original_constraints)

	x1 = x[0]
	x2 = x[1]
	x3 = x[2]
	x4 = x[3]

	P = 6000
	L = 14
	E = 30 * 1e6

	# // deltaMax = 0.25
	G = 12 * 1e6
	tauMax = 13600
	sigmaMax = 30000

	# First original objective function
	f[0] = (1.10471 * x1 * x1 * x2) + (0.04811 * x3 * x4) * (14.0 + x2)
	# Second original objective function
	f[1] = (4 * P * L * L * L) / (E * x4 * x3 * x3 * x3)

	# Constraint functions
	M = P * (L + (x2 / 2))
	tmpVar = ((x2 * x2) / 4.0) + np.power((x1 + x3) / 2.0, 2)
	R = np.sqrt(tmpVar)
	tmpVar = ((x2 * x2) / 12.0) + np.power((x1 + x3) / 2.0, 2)
	J = 2 * np.sqrt(2) * x1 * x2 * tmpVar

	tauDashDash = (M * R) / J    
	tauDash = P / (np.sqrt(2) * x1 * x2)
	tmpVar = tauDash * tauDash + ((2 * tauDash * tauDashDash * x2) / (2 * R)) + (tauDashDash * tauDashDash)
	tau = np.sqrt(tmpVar)
	sigma = (6 * P * L) / (x4 * x3 * x3)
	tmpVar = 4.013 * E * np.sqrt((x3 * x3 * x4 * x4 * x4 * x4 * x4 * x4) / 36.0) / (L * L)
	tmpVar2 = (x3 / (2 * L)) * np.sqrt(E / (4 * G))
	PC = tmpVar * (1 - tmpVar2)

	g[0] = tauMax - tau
	g[1] = sigmaMax - sigma
	g[2] = x4 - x1
	g[3] = PC - P
	g = np.where(g < 0, -g, 0)                
	f[2] = g[0] + g[1] + g[2] + g[3]

	obj = f[0]
	max_f = 326.7815857813289
	min_f = 0.033217020294840964
	return norm_add_noise(obj, max_f, min_f)

def RE32_2(x):
	problem_name = 'RE32'
	n_objectives = 3
	n_variables = 4
	n_constraints = 0
	n_original_constraints = 4
	ubound = np.zeros(n_variables)
	lbound = np.zeros(n_variables)
	lbound[0] = 0.125
	lbound[1] = 0.1
	lbound[2] = 0.1
	lbound[3] = 0.125
	ubound[0] = 5.0
	ubound[1] = 10.0
	ubound[2] = 10.0
	ubound[3] = 5.0
	x = x*(ubound - lbound) + lbound
	f = np.zeros(n_objectives)
	g = np.zeros(n_original_constraints)

	x1 = x[0]
	x2 = x[1]
	x3 = x[2]
	x4 = x[3]

	P = 6000
	L = 14
	E = 30 * 1e6

	# // deltaMax = 0.25
	G = 12 * 1e6
	tauMax = 13600
	sigmaMax = 30000

	# First original objective function
	f[0] = (1.10471 * x1 * x1 * x2) + (0.04811 * x3 * x4) * (14.0 + x2)
	# Second original objective function
	f[1] = (4 * P * L * L * L) / (E * x4 * x3 * x3 * x3)

	# Constraint functions
	M = P * (L + (x2 / 2))
	tmpVar = ((x2 * x2) / 4.0) + np.power((x1 + x3) / 2.0, 2)
	R = np.sqrt(tmpVar)
	tmpVar = ((x2 * x2) / 12.0) + np.power((x1 + x3) / 2.0, 2)
	J = 2 * np.sqrt(2) * x1 * x2 * tmpVar

	tauDashDash = (M * R) / J    
	tauDash = P / (np.sqrt(2) * x1 * x2)
	tmpVar = tauDash * tauDash + ((2 * tauDash * tauDashDash * x2) / (2 * R)) + (tauDashDash * tauDashDash)
	tau = np.sqrt(tmpVar)
	sigma = (6 * P * L) / (x4 * x3 * x3)
	tmpVar = 4.013 * E * np.sqrt((x3 * x3 * x4 * x4 * x4 * x4 * x4 * x4) / 36.0) / (L * L)
	tmpVar2 = (x3 / (2 * L)) * np.sqrt(E / (4 * G))
	PC = tmpVar * (1 - tmpVar2)

	g[0] = tauMax - tau
	g[1] = sigmaMax - sigma
	g[2] = x4 - x1
	g[3] = PC - P
	g = np.where(g < 0, -g, 0)                
	f[2] = g[0] + g[1] + g[2] + g[3]

	max_f = 16722.394750186457
	min_f = 0.0004393632694982018
	obj = f[1]
	return norm_add_noise(obj, max_f, min_f)

def RE32_3(x):
	problem_name = 'RE32'
	n_objectives = 3
	n_variables = 4
	n_constraints = 0
	n_original_constraints = 4
	ubound = np.zeros(n_variables)
	lbound = np.zeros(n_variables)
	lbound[0] = 0.125
	lbound[1] = 0.1
	lbound[2] = 0.1
	lbound[3] = 0.125
	ubound[0] = 5.0
	ubound[1] = 10.0
	ubound[2] = 10.0
	ubound[3] = 5.0
	x = x*(ubound - lbound) + lbound
	f = np.zeros(n_objectives)
	g = np.zeros(n_original_constraints)

	x1 = x[0]
	x2 = x[1]
	x3 = x[2]
	x4 = x[3]

	P = 6000
	L = 14
	E = 30 * 1e6

	# // deltaMax = 0.25
	G = 12 * 1e6
	tauMax = 13600
	sigmaMax = 30000

	# First original objective function
	f[0] = (1.10471 * x1 * x1 * x2) + (0.04811 * x3 * x4) * (14.0 + x2)
	# Second original objective function
	f[1] = (4 * P * L * L * L) / (E * x4 * x3 * x3 * x3)

	# Constraint functions
	M = P * (L + (x2 / 2))
	tmpVar = ((x2 * x2) / 4.0) + np.power((x1 + x3) / 2.0, 2)
	R = np.sqrt(tmpVar)
	tmpVar = ((x2 * x2) / 12.0) + np.power((x1 + x3) / 2.0, 2)
	J = 2 * np.sqrt(2) * x1 * x2 * tmpVar

	tauDashDash = (M * R) / J    
	tauDash = P / (np.sqrt(2) * x1 * x2)
	tmpVar = tauDash * tauDash + ((2 * tauDash * tauDashDash * x2) / (2 * R)) + (tauDashDash * tauDashDash)
	tau = np.sqrt(tmpVar)
	sigma = (6 * P * L) / (x4 * x3 * x3)
	tmpVar = 4.013 * E * np.sqrt((x3 * x3 * x4 * x4 * x4 * x4 * x4 * x4) / 36.0) / (L * L)
	tmpVar2 = (x3 / (2 * L)) * np.sqrt(E / (4 * G))
	PC = tmpVar * (1 - tmpVar2)

	g[0] = tauMax - tau
	g[1] = sigmaMax - sigma
	g[2] = x4 - x1
	g[3] = PC - P
	g = np.where(g < 0, -g, 0)                
	f[2] = g[0] + g[1] + g[2] + g[3]

	max_f = 387075726.8289943
	min_f = 0.0
	obj = f[2]
	return norm_add_noise(obj, max_f, min_f)

def RE33_1(x):
	n_variables = 4
	
	ubound = np.zeros(n_variables)
	lbound = np.zeros(n_variables)
	lbound[0] = 55
	lbound[1] = 75
	lbound[2] = 1000
	lbound[3] = 11
	ubound[0] = 80
	ubound[1] = 110
	ubound[2] = 3000
	ubound[3] = 20
	x = x*(ubound - lbound) + lbound

	x1 = x[0]
	x2 = x[1]
	x4 = x[3]
	
	# First original objective function
	obj = 4.9 * 1e-5 * (x2 * x2 - x1 * x1) * (x4 - 1.0)
	max_f = 8.41203796456161
	min_f = -0.7148953804344631
	return norm_add_noise(obj, max_f, min_f)

def RE33_2(x):
	n_variables = 4
	
	ubound = np.zeros(n_variables)
	lbound = np.zeros(n_variables)
	lbound[0] = 55
	lbound[1] = 75
	lbound[2] = 1000
	lbound[3] = 11
	ubound[0] = 80
	ubound[1] = 110
	ubound[2] = 3000
	ubound[3] = 20
	x = x*(ubound - lbound) + lbound

	x1 = x[0]
	x2 = x[1]
	x3 = x[2]
	x4 = x[3]
	
	# Second original objective function
	obj = ((9.82 * 1e6) * (x2 * x2 - x1 * x1)) / (x3 * x4 * (x2 * x2 * x2 - x1 * x1 * x1))
	max_f = 8.908419030653976
	min_f = 1.1511506608115614
	return norm_add_noise(obj, max_f, min_f)

def RE33_3(x):
	n_variables = 4
	n_original_constraints = 4
	
	ubound = np.zeros(n_variables)
	lbound = np.zeros(n_variables)
	lbound[0] = 55
	lbound[1] = 75
	lbound[2] = 1000
	lbound[3] = 11
	ubound[0] = 80
	ubound[1] = 110
	ubound[2] = 3000
	ubound[3] = 20
	x = x*(ubound - lbound) + lbound
	g = np.zeros(n_original_constraints)

	x1 = x[0]
	x2 = x[1]
	x3 = x[2]
	x4 = x[3]

	# Reformulated objective functions
	g[0] = (x2 - x1) - 20.0
	g[1] = 0.4 - (x3 / (3.14 * (x2 * x2 - x1 * x1)))
	g[2] = 1.0 - (2.22 * 1e-3 * x3 * (x2 * x2 * x2 - x1 * x1 * x1)) / np.power((x2 * x2 - x1 * x1), 2)
	g[3] = (2.66 * 1e-2 * x3 * x4 * (x2 * x2 * x2 - x1 * x1 * x1)) / (x2 * x2 - x1 * x1) - 900.0
	g = np.where(g < 0, -g, 0)
	obj = g[0] + g[1] + g[2] + g[3]
	max_f = 789717.9711891257
	min_f = 0.0
	return norm_add_noise(obj, max_f, min_f)

def RE34_1(x):
	problem_name = 'RE34'
	n_objectives = 3
	n_variables = 5
	n_constraints = 0
	n_original_constraints = 0
	lbound = np.full(n_variables, 1)
	ubound = np.full(n_variables, 3)
	f = np.zeros(n_objectives)
	g = np.zeros(n_original_constraints)
	x = x*(ubound - lbound) + lbound

	x1 = x[0]
	x2 = x[1]
	x3 = x[2]
	x4 = x[3]
	x5 = x[4]

	f[0] = 1640.2823 + (2.3573285 * x1) + (2.3220035 * x2) + (4.5688768 * x3) + (7.7213633 * x4) + (4.4559504 * x5)
	f[1] = 6.5856 + (1.15 * x1) - (1.0427 * x2) + (0.9738 * x3) + (0.8364 * x4) - (0.3695 * x1 * x4) + (0.0861 * x1 * x5) + (0.3628 * x2 * x4)  - (0.1106 * x1 * x1)  - (0.3437 * x3 * x3) + (0.1764 * x4 * x4)
	f[2] = -0.0551 + (0.0181 * x1) + (0.1024 * x2) + (0.0421 * x3) - (0.0073 * x1 * x2) + (0.024 * x2 * x3) - (0.0118 * x2 * x4) - (0.0204 * x3 * x4) - (0.008 * x3 * x5) - (0.0241 * x2 * x2) + (0.0109 * x4 * x4)

	max_f = 1703.90664838227
	min_f = 1662.6440306117302
	obj = f[0]
	return norm_add_noise(obj, max_f, min_f)

def RE34_2(x):
	problem_name = 'RE34'
	n_objectives = 3
	n_variables = 5
	n_constraints = 0
	n_original_constraints = 0
	lbound = np.full(n_variables, 1)
	ubound = np.full(n_variables, 3)
	f = np.zeros(n_objectives)
	g = np.zeros(n_original_constraints)
	x = x*(ubound - lbound) + lbound

	x1 = x[0]
	x2 = x[1]
	x3 = x[2]
	x4 = x[3]
	x5 = x[4]

	f[0] = 1640.2823 + (2.3573285 * x1) + (2.3220035 * x2) + (4.5688768 * x3) + (7.7213633 * x4) + (4.4559504 * x5)
	f[1] = 6.5856 + (1.15 * x1) - (1.0427 * x2) + (0.9738 * x3) + (0.8364 * x4) - (0.3695 * x1 * x4) + (0.0861 * x1 * x5) + (0.3628 * x2 * x4)  - (0.1106 * x1 * x1)  - (0.3437 * x3 * x3) + (0.1764 * x4 * x4)
	f[2] = -0.0551 + (0.0181 * x1) + (0.1024 * x2) + (0.0421 * x3) - (0.0073 * x1 * x2) + (0.024 * x2 * x3) - (0.0118 * x2 * x4) - (0.0204 * x3 * x4) - (0.008 * x3 * x5) - (0.0241 * x2 * x2) + (0.0109 * x4 * x4)

	max_f = 11.696010139266333
	min_f = 6.1900438549263015
	obj = f[1]
	return norm_add_noise(obj, max_f, min_f)
	
def RE34_3(x):
	problem_name = 'RE34'
	n_objectives = 3
	n_variables = 5
	n_constraints = 0
	n_original_constraints = 0
	lbound = np.full(n_variables, 1)
	ubound = np.full(n_variables, 3)
	f = np.zeros(n_objectives)
	g = np.zeros(n_original_constraints)
	x = x*(ubound - lbound) + lbound
	x1 = x[0]
	x2 = x[1]
	x3 = x[2]
	x4 = x[3]
	x5 = x[4]

	f[0] = 1640.2823 + (2.3573285 * x1) + (2.3220035 * x2) + (4.5688768 * x3) + (7.7213633 * x4) + (4.4559504 * x5)
	f[1] = 6.5856 + (1.15 * x1) - (1.0427 * x2) + (0.9738 * x3) + (0.8364 * x4) - (0.3695 * x1 * x4) + (0.0861 * x1 * x5) + (0.3628 * x2 * x4)  - (0.1106 * x1 * x1)  - (0.3437 * x3 * x3) + (0.1764 * x4 * x4)
	f[2] = -0.0551 + (0.0181 * x1) + (0.1024 * x2) + (0.0421 * x3) - (0.0073 * x1 * x2) + (0.024 * x2 * x3) - (0.0118 * x2 * x4) - (0.0204 * x3 * x4) - (0.008 * x3 * x5) - (0.0241 * x2 * x2) + (0.0109 * x4 * x4)

	max_f = 0.2604266470749954
	min_f = 0.04087807037352572
	obj = f[2]
	return norm_add_noise(obj, max_f, min_f)

def RE35_1(x):
	problem_name = 'RE35'
	n_objectives = 3
	n_variables = 7
	n_constraints = 0
	n_original_constraints = 11
	lbound = np.zeros(n_variables)
	ubound = np.zeros(n_variables)
	lbound[0] = 2.6
	lbound[1] = 0.7
	lbound[2] = 17
	lbound[3] = 7.3
	lbound[4] = 7.3
	lbound[5] = 2.9
	lbound[6] = 5.0    
	ubound[0] = 3.6
	ubound[1] = 0.8
	ubound[2] = 28
	ubound[3] = 8.3
	ubound[4] = 8.3
	ubound[5] = 3.9
	ubound[6] = 5.5
	x = x*(ubound - lbound) + lbound
	f = np.zeros(n_objectives)
	g = np.zeros(n_original_constraints)

	x1 = x[0]
	x2 = x[1]
	x3 = np.round(x[2])
	x4 = x[3]
	x5 = x[4]
	x6 = x[5]
	x7 = x[6]

	# First original objective function (weight)
	f[0] = 0.7854 * x1 * (x2 * x2) * (((10.0 * x3 * x3) / 3.0) + (14.933 * x3) - 43.0934) - 1.508 * x1 * (x6 * x6 + x7 * x7) + 7.477 * (x6 * x6 * x6 + x7 * x7 * x7) + 0.7854 * (x4 * x6 * x6 + x5 * x7 * x7)

	# Second original objective function (stress)
	tmpVar = np.power((745.0 * x4) / (x2 * x3), 2.0)  + 1.69 * 1e7
	f[1] =  np.sqrt(tmpVar) / (0.1 * x6 * x6 * x6)

	# Constraint functions 	
	g[0] = -(1.0 / (x1 * x2 * x2 * x3)) + 1.0 / 27.0
	g[1] = -(1.0 / (x1 * x2 * x2 * x3 * x3)) + 1.0 / 397.5
	g[2] = -(x4 * x4 * x4) / (x2 * x3 * x6 * x6 * x6 * x6) + 1.0 / 1.93
	g[3] = -(x5 * x5 * x5) / (x2 * x3 * x7 * x7 * x7 * x7) + 1.0 / 1.93
	g[4] = -(x2 * x3) + 40.0
	g[5] = -(x1 / x2) + 12.0
	g[6] = -5.0 + (x1 / x2)
	g[7] = -1.9 + x4 - 1.5 * x6
	g[8] = -1.9 + x5 - 1.1 * x7
	g[9] =  -f[1] + 1300.0
	tmpVar = np.power((745.0 * x5) / (x2 * x3), 2.0) + 1.575 * 1e8
	g[10] = -np.sqrt(tmpVar) / (0.1 * x7 * x7 * x7) + 1100.0	
	g = np.where(g < 0, -g, 0)                
	f[2] = g[0] + g[1] + g[2] + g[3] + g[4] + g[5] + g[6] + g[7] + g[8] + g[9] + g[10]

	max_f = 7085.019631840887
	min_f = 2390.015417718049
	obj = f[0]
	return norm_add_noise(obj, max_f, min_f)

def RE35_2(x):
	problem_name = 'RE35'
	n_objectives = 3
	n_variables = 7
	n_constraints = 0
	n_original_constraints = 11
	lbound = np.zeros(n_variables)
	ubound = np.zeros(n_variables)
	lbound[0] = 2.6
	lbound[1] = 0.7
	lbound[2] = 17
	lbound[3] = 7.3
	lbound[4] = 7.3
	lbound[5] = 2.9
	lbound[6] = 5.0    
	ubound[0] = 3.6
	ubound[1] = 0.8
	ubound[2] = 28
	ubound[3] = 8.3
	ubound[4] = 8.3
	ubound[5] = 3.9
	ubound[6] = 5.5
	x = x*(ubound - lbound) + lbound
	f = np.zeros(n_objectives)
	g = np.zeros(n_original_constraints)

	x1 = x[0]
	x2 = x[1]
	x3 = np.round(x[2])
	x4 = x[3]
	x5 = x[4]
	x6 = x[5]
	x7 = x[6]

	# First original objective function (weight)
	f[0] = 0.7854 * x1 * (x2 * x2) * (((10.0 * x3 * x3) / 3.0) + (14.933 * x3) - 43.0934) - 1.508 * x1 * (x6 * x6 + x7 * x7) + 7.477 * (x6 * x6 * x6 + x7 * x7 * x7) + 0.7854 * (x4 * x6 * x6 + x5 * x7 * x7)

	# Second original objective function (stress)
	tmpVar = np.power((745.0 * x4) / (x2 * x3), 2.0)  + 1.69 * 1e7
	f[1] =  np.sqrt(tmpVar) / (0.1 * x6 * x6 * x6)

	# Constraint functions 	
	g[0] = -(1.0 / (x1 * x2 * x2 * x3)) + 1.0 / 27.0
	g[1] = -(1.0 / (x1 * x2 * x2 * x3 * x3)) + 1.0 / 397.5
	g[2] = -(x4 * x4 * x4) / (x2 * x3 * x6 * x6 * x6 * x6) + 1.0 / 1.93
	g[3] = -(x5 * x5 * x5) / (x2 * x3 * x7 * x7 * x7 * x7) + 1.0 / 1.93
	g[4] = -(x2 * x3) + 40.0
	g[5] = -(x1 / x2) + 12.0
	g[6] = -5.0 + (x1 / x2)
	g[7] = -1.9 + x4 - 1.5 * x6
	g[8] = -1.9 + x5 - 1.1 * x7
	g[9] =  -f[1] + 1300.0
	tmpVar = np.power((745.0 * x5) / (x2 * x3), 2.0) + 1.575 * 1e8
	g[10] = -np.sqrt(tmpVar) / (0.1 * x7 * x7 * x7) + 1100.0	
	g = np.where(g < 0, -g, 0)                
	f[2] = g[0] + g[1] + g[2] + g[3] + g[4] + g[5] + g[6] + g[7] + g[8] + g[9] + g[10]

	max_f = 1698.401419092365
	min_f = 694.3152735996322
	obj = f[1]
	return norm_add_noise(obj, max_f, min_f)

def RE35_3(x):
	problem_name = 'RE35'
	n_objectives = 3
	n_variables = 7
	n_constraints = 0
	n_original_constraints = 11
	lbound = np.zeros(n_variables)
	ubound = np.zeros(n_variables)
	lbound[0] = 2.6
	lbound[1] = 0.7
	lbound[2] = 17
	lbound[3] = 7.3
	lbound[4] = 7.3
	lbound[5] = 2.9
	lbound[6] = 5.0    
	ubound[0] = 3.6
	ubound[1] = 0.8
	ubound[2] = 28
	ubound[3] = 8.3
	ubound[4] = 8.3
	ubound[5] = 3.9
	ubound[6] = 5.5
	x = x*(ubound - lbound) + lbound
	f = np.zeros(n_objectives)
	g = np.zeros(n_original_constraints)

	x1 = x[0]
	x2 = x[1]
	x3 = np.round(x[2])
	x4 = x[3]
	x5 = x[4]
	x6 = x[5]
	x7 = x[6]

	# First original objective function (weight)
	f[0] = 0.7854 * x1 * (x2 * x2) * (((10.0 * x3 * x3) / 3.0) + (14.933 * x3) - 43.0934) - 1.508 * x1 * (x6 * x6 + x7 * x7) + 7.477 * (x6 * x6 * x6 + x7 * x7 * x7) + 0.7854 * (x4 * x6 * x6 + x5 * x7 * x7)

	# Second original objective function (stress)
	tmpVar = np.power((745.0 * x4) / (x2 * x3), 2.0)  + 1.69 * 1e7
	f[1] =  np.sqrt(tmpVar) / (0.1 * x6 * x6 * x6)

	# Constraint functions 	
	g[0] = -(1.0 / (x1 * x2 * x2 * x3)) + 1.0 / 27.0
	g[1] = -(1.0 / (x1 * x2 * x2 * x3 * x3)) + 1.0 / 397.5
	g[2] = -(x4 * x4 * x4) / (x2 * x3 * x6 * x6 * x6 * x6) + 1.0 / 1.93
	g[3] = -(x5 * x5 * x5) / (x2 * x3 * x7 * x7 * x7 * x7) + 1.0 / 1.93
	g[4] = -(x2 * x3) + 40.0
	g[5] = -(x1 / x2) + 12.0
	g[6] = -5.0 + (x1 / x2)
	g[7] = -1.9 + x4 - 1.5 * x6
	g[8] = -1.9 + x5 - 1.1 * x7
	g[9] =  -f[1] + 1300.0
	tmpVar = np.power((745.0 * x5) / (x2 * x3), 2.0) + 1.575 * 1e8
	g[10] = -np.sqrt(tmpVar) / (0.1 * x7 * x7 * x7) + 1100.0	
	g = np.where(g < 0, -g, 0)                
	f[2] = g[0] + g[1] + g[2] + g[3] + g[4] + g[5] + g[6] + g[7] + g[8] + g[9] + g[10]

	max_f = 400
	min_f = 0.0
	obj = f[2]
	return norm_add_noise(obj, max_f, min_f)

def RE36_1(x):
	problem_name = 'RE36'
	n_objectives = 3
	n_variables = 4
	n_constraints = 0
	n_original_constraints = 1

	lbound = np.full(n_variables, 12)
	ubound = np.full(n_variables, 60)

	f = np.zeros(n_objectives)
	g = np.zeros(n_original_constraints)
	x = x*(ubound - lbound) + lbound

	# all the four variables must be inverger values
	x1 = np.round(x[0])
	x2 = np.round(x[1])
	x3 = np.round(x[2])
	x4 = np.round(x[3])

	# First original objective function
	f[0] = np.abs(6.931 - ((x3 / x1) * (x4 / x2)))
	# Second original objective function (the maximum value among the four variables)
	l = [x1, x2, x3, x4]
	f[1] = max(l)

	g[0] = 0.5 - (f[0] / 6.931)    
	g = np.where(g < 0, -g, 0)                
	f[2] = g[0]

	max_f = 17.242611111111113
	min_f = 0.0
	obj = f[0]
	return norm_add_noise(obj, max_f, min_f)

def RE36_2(x):
	problem_name = 'RE36'
	n_objectives = 3
	n_variables = 4
	n_constraints = 0
	n_original_constraints = 1

	lbound = np.full(n_variables, 12)
	ubound = np.full(n_variables, 60)

	f = np.zeros(n_objectives)
	g = np.zeros(n_original_constraints)
	x = x*(ubound - lbound) + lbound

	# all the four variables must be inverger values
	x1 = np.round(x[0])
	x2 = np.round(x[1])
	x3 = np.round(x[2])
	x4 = np.round(x[3])

	# First original objective function
	f[0] = np.abs(6.931 - ((x3 / x1) * (x4 / x2)))
	# Second original objective function (the maximum value among the four variables)
	l = [x1, x2, x3, x4]
	f[1] = max(l)

	g[0] = 0.5 - (f[0] / 6.931)    
	g = np.where(g < 0, -g, 0)                
	f[2] = g[0]

	max_f = 60.0
	min_f = 13.0
	obj = f[1]
	return norm_add_noise(obj, max_f, min_f)

def RE36_3(x):
	problem_name = 'RE36'
	n_objectives = 3
	n_variables = 4
	n_constraints = 0
	n_original_constraints = 1

	lbound = np.full(n_variables, 12)
	ubound = np.full(n_variables, 60)
	x = x*(ubound - lbound) + lbound
	f = np.zeros(n_objectives)
	g = np.zeros(n_original_constraints)

	# all the four variables must be inverger values
	x1 = np.round(x[0])
	x2 = np.round(x[1])
	x3 = np.round(x[2])
	x4 = np.round(x[3])

	# First original objective function
	f[0] = np.abs(6.931 - ((x3 / x1) * (x4 / x2)))
	# Second original objective function (the maximum value among the four variables)
	l = [x1, x2, x3, x4]
	f[1] = max(l)

	g[0] = 0.5 - (f[0] / 6.931)    
	g = np.where(g < 0, -g, 0)                
	f[2] = g[0]

	max_f = 2
	min_f = 0.0
	obj = f[2]
	return norm_add_noise(obj, max_f, min_f)

def RE37_1(x):
	problem_name = 'RE37'
	n_objectives = 3
	n_variables = 4
	n_constraints = 0
	n_original_constraints = 0

	lbound = np.full(n_variables, 0)
	ubound = np.full(n_variables, 1)

	x = x*(ubound - lbound) + lbound

	f = np.zeros(n_objectives)

	xAlpha = x[0]
	xHA = x[1]
	xOA = x[2]
	xOPTT = x[3]

	# f1 (TF_max)
	f[0] = 0.692 + (0.477 * xAlpha) - (0.687 * xHA) - (0.080 * xOA) - (0.0650 * xOPTT) - (0.167 * xAlpha * xAlpha) - (0.0129 * xHA * xAlpha) + (0.0796 * xHA * xHA) - (0.0634 * xOA * xAlpha) - (0.0257 * xOA * xHA) + (0.0877 * xOA * xOA) - (0.0521 * xOPTT * xAlpha) + (0.00156 * xOPTT * xHA) + (0.00198 * xOPTT * xOA) + (0.0184 * xOPTT * xOPTT)
	# f2 (X_cc)
	f[1] = 0.153 - (0.322 * xAlpha) + (0.396 * xHA) + (0.424 * xOA) + (0.0226 * xOPTT) + (0.175 * xAlpha * xAlpha) + (0.0185 * xHA * xAlpha) - (0.0701 * xHA * xHA) - (0.251 * xOA * xAlpha) + (0.179 * xOA * xHA) + (0.0150 * xOA * xOA) + (0.0134 * xOPTT * xAlpha) + (0.0296 * xOPTT * xHA) + (0.0752 * xOPTT * xOA) + (0.0192 * xOPTT * xOPTT)
	# f3 (TT_max)
	f[2] = 0.370 - (0.205 * xAlpha) + (0.0307 * xHA) + (0.108 * xOA) + (1.019 * xOPTT) - (0.135 * xAlpha * xAlpha) + (0.0141 * xHA * xAlpha) + (0.0998 * xHA * xHA) + (0.208 * xOA * xAlpha) - (0.0301 * xOA * xHA) - (0.226 * xOA * xOA) + (0.353 * xOPTT * xAlpha) - (0.0497 * xOPTT * xOA) - (0.423 * xOPTT * xOPTT) + (0.202 * xHA * xAlpha * xAlpha) - (0.281 * xOA * xAlpha * xAlpha) - (0.342 * xHA * xHA * xAlpha) - (0.245 * xHA * xHA * xOA) + (0.281 * xOA * xOA * xHA) - (0.184 * xOPTT * xOPTT * xAlpha) - (0.281 * xHA * xAlpha * xOA)

	max_f = 1.0
	min_f = 0.0
	obj = f[0]
	return norm_add_noise(obj, max_f, min_f)
	
def RE37_2(x):
	problem_name = 'RE37'
	n_objectives = 3
	n_variables = 4
	n_constraints = 0
	n_original_constraints = 0

	lbound = np.full(n_variables, 0)
	ubound = np.full(n_variables, 1)

	x = x*(ubound - lbound) + lbound

	f = np.zeros(n_objectives)

	xAlpha = x[0]
	xHA = x[1]
	xOA = x[2]
	xOPTT = x[3]

	# f1 (TF_max)
	f[0] = 0.692 + (0.477 * xAlpha) - (0.687 * xHA) - (0.080 * xOA) - (0.0650 * xOPTT) - (0.167 * xAlpha * xAlpha) - (0.0129 * xHA * xAlpha) + (0.0796 * xHA * xHA) - (0.0634 * xOA * xAlpha) - (0.0257 * xOA * xHA) + (0.0877 * xOA * xOA) - (0.0521 * xOPTT * xAlpha) + (0.00156 * xOPTT * xHA) + (0.00198 * xOPTT * xOA) + (0.0184 * xOPTT * xOPTT)
	# f2 (X_cc)
	f[1] = 0.153 - (0.322 * xAlpha) + (0.396 * xHA) + (0.424 * xOA) + (0.0226 * xOPTT) + (0.175 * xAlpha * xAlpha) + (0.0185 * xHA * xAlpha) - (0.0701 * xHA * xHA) - (0.251 * xOA * xAlpha) + (0.179 * xOA * xHA) + (0.0150 * xOA * xOA) + (0.0134 * xOPTT * xAlpha) + (0.0296 * xOPTT * xHA) + (0.0752 * xOPTT * xOA) + (0.0192 * xOPTT * xOPTT)
	# f3 (TT_max)
	f[2] = 0.370 - (0.205 * xAlpha) + (0.0307 * xHA) + (0.108 * xOA) + (1.019 * xOPTT) - (0.135 * xAlpha * xAlpha) + (0.0141 * xHA * xAlpha) + (0.0998 * xHA * xHA) + (0.208 * xOA * xAlpha) - (0.0301 * xOA * xHA) - (0.226 * xOA * xOA) + (0.353 * xOPTT * xAlpha) - (0.0497 * xOPTT * xOA) - (0.423 * xOPTT * xOPTT) + (0.202 * xHA * xAlpha * xAlpha) - (0.281 * xOA * xAlpha * xAlpha) - (0.342 * xHA * xHA * xAlpha) - (0.245 * xHA * xHA * xOA) + (0.281 * xOA * xOA * xHA) - (0.184 * xOPTT * xOPTT * xAlpha) - (0.281 * xHA * xAlpha * xOA)

	max_f = 1.2258253067704867
	min_f = 0.0
	obj = f[1]
	return norm_add_noise(obj, max_f, min_f)
	
def RE37_3(x):
	problem_name = 'RE37'
	n_objectives = 3
	n_variables = 4
	n_constraints = 0
	n_original_constraints = 0

	lbound = np.full(n_variables, 0)
	ubound = np.full(n_variables, 1)

	x = x*(ubound - lbound) + lbound

	f = np.zeros(n_objectives)

	xAlpha = x[0]
	xHA = x[1]
	xOA = x[2]
	xOPTT = x[3]

	# f1 (TF_max)
	f[0] = 0.692 + (0.477 * xAlpha) - (0.687 * xHA) - (0.080 * xOA) - (0.0650 * xOPTT) - (0.167 * xAlpha * xAlpha) - (0.0129 * xHA * xAlpha) + (0.0796 * xHA * xHA) - (0.0634 * xOA * xAlpha) - (0.0257 * xOA * xHA) + (0.0877 * xOA * xOA) - (0.0521 * xOPTT * xAlpha) + (0.00156 * xOPTT * xHA) + (0.00198 * xOPTT * xOA) + (0.0184 * xOPTT * xOPTT)
	# f2 (X_cc)
	f[1] = 0.153 - (0.322 * xAlpha) + (0.396 * xHA) + (0.424 * xOA) + (0.0226 * xOPTT) + (0.175 * xAlpha * xAlpha) + (0.0185 * xHA * xAlpha) - (0.0701 * xHA * xHA) - (0.251 * xOA * xAlpha) + (0.179 * xOA * xHA) + (0.0150 * xOA * xOA) + (0.0134 * xOPTT * xAlpha) + (0.0296 * xOPTT * xHA) + (0.0752 * xOPTT * xOA) + (0.0192 * xOPTT * xOPTT)
	# f3 (TT_max)
	f[2] = 0.370 - (0.205 * xAlpha) + (0.0307 * xHA) + (0.108 * xOA) + (1.019 * xOPTT) - (0.135 * xAlpha * xAlpha) + (0.0141 * xHA * xAlpha) + (0.0998 * xHA * xHA) + (0.208 * xOA * xAlpha) - (0.0301 * xOA * xHA) - (0.226 * xOA * xOA) + (0.353 * xOPTT * xAlpha) - (0.0497 * xOPTT * xOA) - (0.423 * xOPTT * xOPTT) + (0.202 * xHA * xAlpha * xAlpha) - (0.281 * xOA * xAlpha * xAlpha) - (0.342 * xHA * xHA * xAlpha) - (0.245 * xHA * xHA * xOA) + (0.281 * xOA * xOA * xHA) - (0.184 * xOPTT * xOPTT * xAlpha) - (0.281 * xHA * xAlpha * xOA)

	max_f = 1.0850670780811584
	min_f = -0.4081426955480145
	obj = f[2]
	return norm_add_noise(obj, max_f, min_f)

def RE21_1(x):
	problem_name = 'RE21'
	n_objectives = 2
	n_variables = 4
	n_constraints = 0        
	n_original_constraints = 0
	
	F = 10.0
	sigma = 10.0
	tmp_val = F / sigma

	ubound = np.full(n_variables, 3 * tmp_val)
	lbound = np.zeros(n_variables)
	lbound[0] = tmp_val
	lbound[1] = np.sqrt(2.0) * tmp_val
	lbound[2] = np.sqrt(2.0) * tmp_val
	lbound[3] = tmp_val
	x = x*(ubound - lbound) + lbound
	f = np.zeros(n_objectives)
	x1 = x[0]
	x2 = x[1]
	x3 = x[2]
	x4 = x[3]

	F = 10.0
	sigma = 10.0
	E = 2.0 * 1e5
	L = 200.0

	f[0] = L * ((2 * x1) + np.sqrt(2.0) * x2 + np.sqrt(x3) + x4)
	f[1] = ((F * L) / E) * ((2.0 / x1) + (2.0 * np.sqrt(2.0) / x2) - (2.0 * np.sqrt(2.0) / x3) + (2.0 / x4))

	max_f = 2977.35881731081 
	min_f = 1257.2744755078438
	obj = f[0]
	return norm_add_noise(obj, max_f, min_f)

def RE21_2(x):
	problem_name = 'RE21'
	n_objectives = 2
	n_variables = 4
	n_constraints = 0        
	n_original_constraints = 0
	
	F = 10.0
	sigma = 10.0
	tmp_val = F / sigma

	ubound = np.full(n_variables, 3 * tmp_val)
	lbound = np.zeros(n_variables)
	lbound[0] = tmp_val
	lbound[1] = np.sqrt(2.0) * tmp_val
	lbound[2] = np.sqrt(2.0) * tmp_val
	lbound[3] = tmp_val
	x = x*(ubound - lbound) + lbound
	f = np.zeros(n_objectives)
	x1 = x[0]
	x2 = x[1]
	x3 = x[2]
	x4 = x[3]

	F = 10.0
	sigma = 10.0
	E = 2.0 * 1e5
	L = 200.0

	f[0] = L * ((2 * x1) + np.sqrt(2.0) * x2 + np.sqrt(x3) + x4)
	f[1] = ((F * L) / E) * ((2.0 / x1) + (2.0 * np.sqrt(2.0) / x2) - (2.0 * np.sqrt(2.0) / x3) + (2.0 / x4))
	
	max_f = 0.04997689402824656
	min_f = 0.0029771351751672216
	obj = f[1]
	return norm_add_noise(obj, max_f, min_f)

def RE22_1(x):
	problem_name = 'RE22'
	n_objectives = 2
	n_variables = 3
	n_constraints = 0
	n_original_constraints = 2
	ubound = np.zeros(n_variables)
	lbound = np.zeros(n_variables)
	lbound[0] = 0.2
	lbound[1] = 0.1
	lbound[2] = 0.1
	ubound[0] = 15
	ubound[1] = 20
	ubound[2] = 40
	x = x*(ubound - lbound) + lbound
	feasible_vals = np.array([0.20, 0.31, 0.40, 0.44, 0.60, 0.62, 0.79, 0.80, 0.88, 0.93, 1.0, 1.20, 1.24, 1.32, 1.40, 1.55, 1.58, 1.60, 1.76, 1.80, 1.86, 2.0, 2.17, 2.20, 2.37, 2.40, 2.48, 2.60, 2.64, 2.79, 2.80, 3.0, 3.08, 3,10, 3.16, 3.41, 3.52, 3.60, 3.72, 3.95, 3.96, 4.0, 4.03, 4.20, 4.34, 4.40, 4.65, 4.74, 4.80, 4.84, 5.0, 5.28, 5.40, 5.53, 5.72, 6.0, 6.16, 6.32, 6.60, 7.11, 7.20, 7.80, 7.90, 8.0, 8.40, 8.69, 9.0, 9.48, 10.27, 11.0, 11.06, 11.85, 12.0, 13.0, 14.0, 15.0])

	f = np.zeros(n_objectives)
	g = np.zeros(n_original_constraints)
	#Reference: getNearestValue_sample2.py (https://gist.github.com/icchi-h/1d0bb1c52ebfdd31f14b3e811328390a)
	idx = np.abs(np.asarray(feasible_vals) - x[0]).argmin()
	x1 = feasible_vals[idx]
	x2 = x[1]
	x3 = x[2]

	#First original objective function
	f[0] = (29.4 * x1) + (0.6 * x2 * x3)

	# Original constraint functions 	
	g[0] = (x1 * x3) - 7.735 * ((x1 * x1) / x2) - 180.0
	g[1] = 4.0 - (x3 / x2)
	g = np.where(g < 0, -g, 0)          
	f[1] = g[0] + g[1]
	
	max_f = 920.0528195769679
	min_f = 5.885428538601677
	obj = f[0]
	return norm_add_noise(obj, max_f, min_f)

def RE22_2(x):
	problem_name = 'RE22'
	n_objectives = 2
	n_variables = 3
	n_constraints = 0
	n_original_constraints = 2
	ubound = np.zeros(n_variables)
	lbound = np.zeros(n_variables)
	lbound[0] = 0.2
	lbound[1] = 0.1
	lbound[2] = 0.1
	ubound[0] = 15
	ubound[1] = 20
	ubound[2] = 40
	x = x*(ubound - lbound) + lbound
	feasible_vals = np.array([0.20, 0.31, 0.40, 0.44, 0.60, 0.62, 0.79, 0.80, 0.88, 0.93, 1.0, 1.20, 1.24, 1.32, 1.40, 1.55, 1.58, 1.60, 1.76, 1.80, 1.86, 2.0, 2.17, 2.20, 2.37, 2.40, 2.48, 2.60, 2.64, 2.79, 2.80, 3.0, 3.08, 3,10, 3.16, 3.41, 3.52, 3.60, 3.72, 3.95, 3.96, 4.0, 4.03, 4.20, 4.34, 4.40, 4.65, 4.74, 4.80, 4.84, 5.0, 5.28, 5.40, 5.53, 5.72, 6.0, 6.16, 6.32, 6.60, 7.11, 7.20, 7.80, 7.90, 8.0, 8.40, 8.69, 9.0, 9.48, 10.27, 11.0, 11.06, 11.85, 12.0, 13.0, 14.0, 15.0])

	f = np.zeros(n_objectives)
	g = np.zeros(n_original_constraints)
	#Reference: getNearestValue_sample2.py (https://gist.github.com/icchi-h/1d0bb1c52ebfdd31f14b3e811328390a)
	idx = np.abs(np.asarray(feasible_vals) - x[0]).argmin()
	x1 = feasible_vals[idx]
	x2 = x[1]
	x3 = x[2]

	#First original objective function
	f[0] = (29.4 * x1) + (0.6 * x2 * x3)

	# Original constraint functions 	
	g[0] = (x1 * x3) - 7.735 * ((x1 * x1) / x2) - 180.0
	g[1] = 4.0 - (x3 / x2)
	g = np.where(g < 0, -g, g)          
	f[1] = g[0] + g[1]

	max_f = 17430.35713212821
	min_f = 0
	obj = f[1]
	return norm_add_noise(obj, max_f, min_f)

def RE23_1(x):
	problem_name = 'RE23'
	n_objectives = 2
	n_variables = 4
	n_constraints = 0
	n_original_constraints = 3
	ubound = np.zeros(n_variables)
	lbound = np.zeros(n_variables)
	lbound[0] = 1
	lbound[1] = 1
	lbound[2] = 10
	lbound[3] = 10
	ubound[0] = 100
	ubound[1] = 100
	ubound[2] = 200
	ubound[3] = 240
	x = x*(ubound - lbound) + lbound
	f = np.zeros(n_objectives)
	g = np.zeros(n_original_constraints)

	x1 = 0.0625 * int(np.round(x[0]))
	x2 = 0.0625 * int(np.round(x[1]))        
	x3 = x[2]
	x4 = x[3]

	#First original objective function
	f[0] = (0.6224 * x1 * x3* x4) + (1.7781 * x2 * x3 * x3) + (3.1661 * x1 * x1 * x4) + (19.84 * x1 * x1 * x3)

	# Original constraint functions 	
	g[0] = x1 - (0.0193 * x3)
	g[1] = x2 - (0.00954 * x3)
	g[2] = (np.pi * x3 * x3 * x4) + ((4.0/3.0) * (np.pi * x3 * x3 * x3)) - 1296000
	g = np.where(g < 0, -g, 0)            
	f[1] = g[0] + g[1] + g[2]

	max_f = 803948.4670560489
	min_f = 46.10361249611967
	obj = f[0]
	return norm_add_noise(obj, max_f, min_f)

def RE23_2(x):
	problem_name = 'RE23'
	n_objectives = 2
	n_variables = 4
	n_constraints = 0
	n_original_constraints = 3
	ubound = np.zeros(n_variables)
	lbound = np.zeros(n_variables)
	lbound[0] = 1
	lbound[1] = 1
	lbound[2] = 10
	lbound[3] = 10
	ubound[0] = 100
	ubound[1] = 100
	ubound[2] = 200
	ubound[3] = 240
	x = x*(ubound - lbound) + lbound
	f = np.zeros(n_objectives)
	g = np.zeros(n_original_constraints)

	x1 = 0.0625 * int(np.round(x[0]))
	x2 = 0.0625 * int(np.round(x[1]))        
	x3 = x[2]
	x4 = x[3]

	#First original objective function
	f[0] = (0.6224 * x1 * x3* x4) + (1.7781 * x2 * x3 * x3) + (3.1661 * x1 * x1 * x4) + (19.84 * x1 * x1 * x3)

	# Original constraint functions 	
	g[0] = x1 - (0.0193 * x3)
	g[1] = x2 - (0.00954 * x3)
	g[2] = (np.pi * x3 * x3 * x4) + ((4.0/3.0) * (np.pi * x3 * x3 * x3)) - 1296000
	g = np.where(g < 0, -g, 0)            
	f[1] = g[0] + g[1] + g[2]

	max_f = 1288593.9809127934
	min_f = 0.0
	obj = f[1]
	return norm_add_noise(obj, max_f, min_f)

def RE24_1(x):
	problem_name = 'RE24'
	n_objectives = 2
	n_variables = 2
	n_constraints = 0
	n_original_constraints = 4
	ubound = np.zeros(n_variables)
	lbound = np.zeros(n_variables)
	lbound[0] = 0.5
	lbound[1] = 0.5
	ubound[0] = 4
	ubound[1] = 50
	x = x*(ubound - lbound) + lbound
	f = np.zeros(n_objectives)
	g = np.zeros(n_original_constraints)

	x1 = x[0]
	x2 = x[1]

	#First original objective function
	f[0] = x1 + (120 * x2)

	E = 700000
	sigma_b_max = 700
	tau_max = 450
	delta_max = 1.5
	sigma_k = (E * x1 * x1) / 100
	sigma_b = 4500 / (x1 * x2)
	tau = 1800 / x2
	delta = (56.2 * 10000) / (E * x1 * x2 * x2)

	g[0] = 1 - (sigma_b / sigma_b_max)
	g[1] = 1 - (tau / tau_max)
	g[2] = 1 - (delta / delta_max)
	g[3] = 1 - (sigma_b / sigma_k)
	g = np.where(g < 0, -g, 0)            
	f[1] = g[0] + g[1] + g[2] + g[3]

	max_f = 6003.972054541111
	min_f = 60.57111063599586
	obj = f[0]
	return norm_add_noise(obj, max_f, min_f)

def RE24_2(x):
	problem_name = 'RE24'
	n_objectives = 2
	n_variables = 2
	n_constraints = 0
	n_original_constraints = 4
	ubound = np.zeros(n_variables)
	lbound = np.zeros(n_variables)
	lbound[0] = 0.5
	lbound[1] = 0.5
	ubound[0] = 4
	ubound[1] = 50
	x = x*(ubound - lbound) + lbound
	f = np.zeros(n_objectives)
	g = np.zeros(n_original_constraints)

	x1 = x[0]
	x2 = x[1]

	#First original objective function
	f[0] = x1 + (120 * x2)

	E = 700000
	sigma_b_max = 700
	tau_max = 450
	delta_max = 1.5
	sigma_k = (E * x1 * x1) / 100
	sigma_b = 4500 / (x1 * x2)
	tau = 1800 / x2
	delta = (56.2 * 10000) / (E * x1 * x2 * x2)

	g[0] = 1 - (sigma_b / sigma_b_max)
	g[1] = 1 - (tau / tau_max)
	g[2] = 1 - (delta / delta_max)
	g[3] = 1 - (sigma_b / sigma_k)
	g = np.where(g < 0, -g, 0)            
	f[1] = g[0] + g[1] + g[2] + g[3]

	max_f = 41.569790502100105
	min_f = 0
	obj = f[1]
	return norm_add_noise(obj, max_f, min_f)

def RE25_1(x):
	
	n_variables = 3
	
	ubound = np.zeros(n_variables)
	lbound = np.zeros(n_variables)
	lbound[0] = 1
	lbound[1] = 0.6
	lbound[2] = 0.09
	ubound[0] = 70
	ubound[1] = 3
	ubound[2] = 0.5
	
	feasible_vals = np.array([0.009, 0.0095, 0.0104, 0.0118, 0.0128, 0.0132, 0.014, 0.015, 0.0162, 0.0173, 0.018, 0.02, 0.023, 0.025, 0.028, 0.032, 0.035, 0.041, 0.047, 0.054, 0.063, 0.072, 0.08, 0.092, 0.105, 0.12, 0.135, 0.148, 0.162, 0.177, 0.192, 0.207, 0.225, 0.244, 0.263, 0.283, 0.307, 0.331, 0.362, 0.394, 0.4375, 0.5])

	x = x*(ubound - lbound) + lbound

	x1 = np.round(x[0])
	x2 = x[1]
	#Reference: getNearestValue_sample2.py (https://gist.github.com/icchi-h/1d0bb1c52ebfdd31f14b3e811328390a)
	idx = np.abs(np.asarray(feasible_vals) - x[2]).argmin()
	x3 = feasible_vals[idx]

	# first original objective function
	obj = (np.pi * np.pi * x2 * x3 * x3 * (x1 + 2)) / 4.0
 
	max_f = 133.22668584938896
	min_f = 0.03776961289196103
	return norm_add_noise(obj, max_f, min_f)
	
def RE25_2(x):
	n_objectives = 2
	n_variables = 3
	n_original_constraints = 6
	
	ubound = np.zeros(n_variables)
	lbound = np.zeros(n_variables)
	lbound[0] = 1
	lbound[1] = 0.6
	lbound[2] = 0.09
	ubound[0] = 70
	ubound[1] = 3
	ubound[2] = 0.5
	x = x*(ubound - lbound) + lbound
	feasible_vals = np.array([0.009, 0.0095, 0.0104, 0.0118, 0.0128, 0.0132, 0.014, 0.015, 0.0162, 0.0173, 0.018, 0.02, 0.023, 0.025, 0.028, 0.032, 0.035, 0.041, 0.047, 0.054, 0.063, 0.072, 0.08, 0.092, 0.105, 0.12, 0.135, 0.148, 0.162, 0.177, 0.192, 0.207, 0.225, 0.244, 0.263, 0.283, 0.307, 0.331, 0.362, 0.394, 0.4375, 0.5])
	g = np.zeros(n_original_constraints)

	x1 = np.round(x[0])
	x2 = x[1]
	#Reference: getNearestValue_sample2.py (https://gist.github.com/icchi-h/1d0bb1c52ebfdd31f14b3e811328390a)
	idx = np.abs(np.asarray(feasible_vals) - x[2]).argmin()
	x3 = feasible_vals[idx]
	
	# constraint functions
	Cf = ((4.0 * (x2 / x3) - 1) / (4.0 * (x2 / x3) - 4)) + (0.615 * x3 / x2)
	Fmax = 1000.0
	S = 189000.0
	G = 11.5 * 1e+6
	K  = (G * x3 * x3 * x3 * x3) / (8 * x1 * x2 * x2 * x2)
	lmax = 14.0
	lf = (Fmax / K) + 1.05 *  (x1 + 2) * x3
	Fp = 300.0
	sigmaP = Fp / K
	sigmaPM = 6
	sigmaW = 1.25

	g[0] = -((8 * Cf * Fmax * x2) / (np.pi * x3 * x3 * x3)) + S
	g[1] = -lf + lmax
	g[2] = -3 + (x2 / x3)
	g[3] = -sigmaP + sigmaPM
	g[4] = -sigmaP - ((Fmax - Fp) / K) - 1.05 * (x1 + 2) * x3 + lf
	g[5] = sigmaW- ((Fmax - Fp) / K)

	g = np.where(g < 0, -g, 0)            
	obj = g[0] + g[1] + g[2] + g[3] + g[4] + g[5]  
 
	max_f = 10074688.939724587
	min_f = 0.0
	return norm_add_noise(obj, max_f, min_f)

scene = "lego"
instance = "6838708"
def set_NERF_scene(x, ins="6838708"):
	# synthetic:  'lego', 'materials', 'mic', 'ship', 'chairs'
	global scene, instance, df_synthetic
	if x == "chairs":
		scene = x
		instance = ins
		df_synthetic = pandas.read_csv(f'./Environment/chairs/all/{instance}.csv')
		df_synthetic = df_synthetic.drop('dataset', axis = 1)
		df_synthetic = df_synthetic.drop('ssim', axis = 1)
		df_synthetic = df_synthetic.drop('lpips', axis = 1)
	else:
		scene = x

def get_NERF_scene():
	# synthetic:  'lego', 'materials', 'mic', 'ship', 'chairs'
	global scene, instance
	return scene, instance

df_synthetic = pandas.read_csv('./Environment/record_synthetic_v2.csv')

# clean data
df_synthetic = df_synthetic.drop('dataset', axis = 1)
df_synthetic = df_synthetic.drop('ssim', axis = 1)
df_synthetic = df_synthetic.drop('lpips', axis = 1)
# df_synthetic = df_synthetic.drop('npt', axis = 1)

def NERF_synthetic_1(x):
	global scene
	df = df_synthetic
	if scene != "chairs":
		df = df[df['scene'] == scene]
	df = df.drop('scene', axis = 1)
	df = df.drop('size', axis = 1)
	df = df.drop('npt', axis = 1)

	# normalize data
	df=(df-df.min())/(df.max()-df.min())
	if scene == "chairs":
		df = df.fillna(0)

	# get all y
	y = df['psnr']
	df = df.drop('psnr', axis = 1)

	# find row
	df = df - x
	df['norm'] = df.apply(np.linalg.norm, axis=1)
	min_norm_index = df['norm'].idxmin()
	y = y.loc[min_norm_index]
	return norm_add_noise(y, 1, 0)

def NERF_synthetic_2(x):
	global scene
	df = df_synthetic
	if scene != "chairs":
		df = df[df['scene'] == scene]
	df = df.drop('scene', axis = 1)
	df = df.drop('psnr', axis = 1)
	df = df.drop('npt', axis = 1)

	# normalize data
	df=(df-df.min())/(df.max()-df.min())
	if scene == "chairs":
		df = df.fillna(0)
	# get all y
	y = -df['size'] + 1
	df = df.drop('size', axis = 1)

	# find row
	df = df - x
	df['norm'] = df.apply(np.linalg.norm, axis=1)
	min_norm_index = df['norm'].idxmin()
	y = y.loc[min_norm_index]
	return norm_add_noise(y, 1, 0)

def NERF_synthetic_3(x):
	global scene
	df = df_synthetic
	if scene != "chairs":
		df = df[df['scene'] == scene]
	df = df.drop('scene', axis = 1)
	df = df.drop('size', axis = 1)
	df = df.drop('psnr', axis = 1)

	# normalize data
	df=(df-df.min())/(df.max()-df.min())
	if scene == "chairs":
		df = df.fillna(0)
	# get all y
	y = -df['npt'] + 1
	df = df.drop('npt', axis = 1)

	# find row
	df = df - x
	df['norm'] = df.apply(np.linalg.norm, axis=1)
	min_norm_index = df['norm'].idxmin()
	y = y.loc[min_norm_index]
	return norm_add_noise(y, 1, 0)

if __name__ == '__main__':
	_, _, mx, mn = optimization_function('NERF', NERF_synthetic_2, 1000, 0)
	print(mx, mn)
	
