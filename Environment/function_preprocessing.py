import numpy as np
from pymoo.factory import get_performance_indicator
from pymoo.config import Config
import sobol_seq
from benchmark_functions import Branin, Currin, Ackley, Rosen, Sphere, Dixon, Rastrigin, Zakharov, Schwefel
from benchmark_functions import GP_function
from benchmark_functions import RE21_1, RE21_2, RE22_1, RE22_2, RE23_1, RE23_2, RE24_1, RE24_2, RE25_1, RE25_2
from benchmark_functions import RE31_1, RE31_2, RE32_1, RE32_2, RE33_1, RE33_2, RE33_3, RE34_1, RE34_2, RE35_1, RE35_2, RE36_1, RE36_2, RE37_1, RE37_2
from benchmark_functions import RE31_3, RE32_3, RE34_3, RE35_3, RE36_3, RE37_3
from benchmark_functions import NERF_synthetic_1, NERF_synthetic_2, NERF_synthetic_3, get_NERF_scene
import torch
import random
import pandas
import numpy as np
import sys, os
import torch
from botorch import fit_gpytorch_mll
from botorch.models import SingleTaskGP, ModelListGP
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import RBFKernel, ScaleKernel, MaternKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import SumMarginalLogLikelihood
    
def optimization_function(f, x):
    dim = np.shape(x)[1]
    y = np.array([f(xx) for xx in x])
    train_X = torch.tensor(x, dtype=float)
    train_Y = torch.tensor(y, dtype=float)
    gp = SingleTaskGP(train_X=train_X, train_Y=train_Y.unsqueeze(1), 
                                outcome_transform=Standardize(1))
    torch.tensor([[0.001]*dim])
    gp.covar_module.base_kernel.lengthscale = torch.tensor([[0.001]*dim]) # important step that make the learned ls be correct
    # model.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-5))
    mll = ExactMarginalLogLikelihood(likelihood=gp.likelihood, model=gp)
    try:
        fit_gpytorch_mll(mll)
    except RuntimeError:
        print("Something wrong")
    return list(gp.covar_module.base_kernel.lengthscale.detach().numpy().squeeze())



def getFuntion(X, f_num, function_type, dim = 1):
	kernels = []
	kernel_lss = []
	if (function_type == "train_05" or function_type == "train_005" or
	 	function_type == "train" or function_type == "train_large" or
	 	function_type == "RBF_0.05" or function_type == "RBF_0.2" or 
		function_type == "matern52_0.05" or function_type == "matern52_0.2" or 
		function_type == "RBF_0.05_large" or function_type == "matern52_0.05_large"):
		f = []
		
		for _ in range(f_num):
			F, kernel, kernel_ls = GP_function(X, function_type, dim = dim)
			kernels.append(kernel)
			kernel_lss.append(kernel_ls)
			f.append(F)
	elif (function_type == "BC"):
		f = [Branin, Currin]
	elif (function_type == "AR"):
		f = [Ackley, Rosen]
	elif (function_type == "ARa"):
		f = [Ackley, Rastrigin]
	elif (function_type == "DR"):
		f = [Dixon, Rastrigin]
	elif (function_type == "ARS"):
		f = [Ackley, Rosen, Sphere]
	elif (function_type == "ABC"):
		f = [Ackley, Branin, Currin]
	elif (function_type == "ABD"):
		f = [Ackley, Branin, Dixon]
	elif (function_type == "ACD"):
		f = [Ackley, Currin, Dixon]
	elif (function_type == "BCD"):
		f = [Branin, Currin, Dixon]
	elif (function_type == "ASR"):
		f = [Ackley, Schwefel, Rastrigin]
	elif (function_type == "DRZ"):
		f = [Dixon, Rastrigin, Zakharov]
	elif (function_type == "Branin"):
		f = [Branin]
	elif (function_type == "Currin"):
		f = [Currin]
	elif (function_type == "RE21"): # Four bar truss design
		f = [RE21_1, RE21_2]
	elif (function_type == "RE22"): # Reinforced concrete beam design
		f = [RE22_1, RE22_2]
	elif (function_type == "RE23"): # Pressure vessel design
		f = [RE23_1, RE23_2]
	elif (function_type == "RE24"): # Hatch cover design 
		f = [RE24_1, RE24_2]
	elif (function_type == "RE25"): # Coil compression spring design
		f = [RE25_1, RE25_2]
	elif (function_type == "RE31"): # Two bar truss design
		f = [RE31_1, RE31_2, RE31_3]
	elif (function_type == "RE32"): # Welded beam design
		f = [RE32_1, RE32_2, RE32_3]
	elif (function_type == "RE33"): # Disc brake design 
		f = [RE33_1, RE33_2, RE33_3]
	elif (function_type == "RE34"): # Vehicle crashworthiness design 
		f = [RE34_1, RE34_2, RE34_3]
	elif (function_type == "RE35"): # Speed reducer design
		f = [RE35_1, RE35_2, RE35_3]
	elif (function_type == "RE36"): # Gear train design 
		f = [RE36_1, RE36_2, RE36_3]
	elif (function_type == "RE37"): # Rocket injector design
		f = [RE37_1, RE37_2, RE37_3]
	elif (function_type == "NERF_synthetic"):
		f = [NERF_synthetic_1, NERF_synthetic_2]
	elif (function_type == "NERF_synthetic_fnum_3"):
		f = [NERF_synthetic_1, NERF_synthetic_2, NERF_synthetic_3]

	# if (function_type != "train_05" and function_type != "train_005" and
	# 	function_type != "train" and function_type != "train_large" and
	#  	function_type != "RBF_0.05" and function_type != "RBF_0.2" and 
	# 	function_type != "matern52_0.05" and function_type != "matern52_0.2"):
	# 	for ff in f:
	# 		kernel_lss.append(optimization_function(ff, domain(function_type, 1000, seed = 0)))
	if (function_type == "BC"):
			kernel_lss = [[1.2322024741645445, 3.473200049038078],
                     		[0.9409959672834756, 1.6117729643698377]]
	elif (function_type == "AR"):	
		kernel_lss = [[0.1067406142139499, 0.10490022738119344],
						[1.1737748968055879, 4.516103731528372]]
	elif (function_type == "ARa"):	
		kernel_lss = [[0.1067406142139499, 0.10490022738119344],
						[0.06388384531435684, 0.06386143410111994]]
	elif (function_type == "DR"):
		kernel_lss = [[8.127061999241256, 0.6611835034260138],
						[0.06388384531435684, 0.06386143410111994]]	
	elif (function_type == "ARS"):
		kernel_lss = [[0.1067406142139499, 0.10490022738119344],
						[1.1737748968055879, 4.516103731528372],
						[2.949756542381238, 2.948504879565341]]
	elif (function_type == "ASR"):
		kernel_lss = [[0.1067406142139499, 0.10490022738119344],
						[0.20099197734756447, 0.2008937879613179],
						[0.06388384531435684, 0.06386143410111994]]
	elif (function_type == "ABC"):
		kernel_lss = [[0.1067406142139499, 0.10490022738119344],
						[1.2322024741645445, 3.473200049038078],
                     	[0.9409959672834756, 1.6117729643698377]]
	elif (function_type == "ABD"):
		kernel_lss = [[0.1067406142139499, 0.10490022738119344],
						[1.2322024741645445, 3.473200049038078],
						[8.127061999241256, 0.6611835034260138]]
	elif (function_type == "ACD"):
		kernel_lss = [[0.1067406142139499, 0.10490022738119344],
						[0.9409959672834756, 1.6117729643698377],
						[8.127061999241256, 0.6611835034260138]]
	elif (function_type == "BCD"):
		kernel_lss = [[1.2322024741645445, 3.473200049038078],
						[0.9409959672834756, 1.6117729643698377],
						[8.127061999241256, 0.6611835034260138]]
	elif (function_type == "DRZ"):
		kernel_lss = [[8.127061999241256, 0.6611835034260138],
						[0.06388384531435684, 0.06386143410111994],
						[2.9270276255879915, 1.6228369089285533]]
	elif (function_type == "RE31"):
		kernel_lss = [[4.2089474135408365, 4.201065221954299, 4.200656582106832],
						[0.02195962318566807, 19.780389409121224, 5.504329936667641],
						[4.208947418704599, 4.201065227061116, 4.20065658717857]]
	elif (function_type == "RE32"):
		kernel_lss = [[3.1247732503449233, 5.015199705183447, 8.349677430066079, 8.36452511770429],
						[18.591385240214237, 10.957533852609469, 0.04095842656505128, 1.6587118917369543],
						[19.239260677128303, 15.077850615899044, 0.07076275625074903, 1.109776752360057]]
	elif (function_type == "RE33"):
		kernel_lss = [[5.20475761317857, 4.694354140980532, 5.404591328251455, 4.903970767934206],
						[7.086579091961371, 7.004084804020621, 2.472594443982033, 6.212144213752191],
						[0.32777215192249776, 0.09090844463579181, 1.2394885530125332, 2.6927242892120846]]
	elif (function_type == "RE34"):
		kernel_lss = [[7.235940329948041, 7.262771512942976, 6.641527441327613, 5.174757305271093, 6.720647843780357],
						[5.764238688285515, 6.106212115083782, 3.704405513576522, 2.7199605891461363, 7.005837182633211],
						[7.931191447563538, 2.994072816658113, 4.3307416184259475, 4.565356636651643, 7.851665661345994]]
	elif (function_type == "RE35"):
		kernel_lss = [[2.392152212944576, 2.6125407681563257, 0.9107941100560183, 2.4691777917093543, 3.8258736573211656, 2.2215775482399027, 3.5778035683484437],
						[6.615963919787311, 6.60388773244275, 6.612403386176171, 6.62149564301235, 6.6118735778871125, 0.7458584451507319, 6.616391053573797],
						[8.094552332284318, 8.036434156103445, 7.8252575412267875, 8.01850655967469, 8.027031665413904, 0.14805068007318284, 8.052223895604318]]
	elif (function_type == "RE36"):
		kernel_lss = [[0.5948346466424413, 0.6024491240860617, 1.3671467792285528, 0.961451786504103],
						[0.6357172256643281, 0.660310643626246, 0.6231943912047118, 0.6183437833044475],
						[0.951523555102253, 0.7852109616805245, 1.2274158894957186, 0.8547764845214822]]
	elif (function_type == "RE37"):
		kernel_lss = [[4.799539862895141, 4.832471686630555, 5.1104207720335095, 5.20329524330567],
						[4.801657799846086, 5.027262909053975, 4.898586784538537, 5.285738391435452],
						[4.185088621143237, 4.028837107371882, 4.669157143722919, 4.529845076599971]]
	elif (function_type == "RE21"):
		kernel_lss = [[3.803722258485652, 5.374228213666235, 6.025312754743776, 5.528794804317436],
						[3.3095155184187006, 4.918713083497343, 5.067419704425443, 3.34007851301534]]
	elif (function_type == "RE22"):
		kernel_lss = [[3.608664369871488, 2.6126497127464767, 2.4039663374248716],
						[0.5785804144220766, 0.015867825479766763, 10.55302741416124]]
	elif (function_type == "RE23"):
		kernel_lss = [[4.122225232973132, 4.817929466403503, 2.118850175864754, 6.977704684655366],
						[10.425123483917059, 10.869433601061608, 0.09074410062298432, 0.5881600281686659]]
	elif (function_type == "RE24"):
		kernel_lss = [[3.5654943048587073, 3.5653157332215546],
						[0.0010919092771883466, 0.00109190927707247]]
	elif (function_type == "RE25"):
		kernel_lss = [[4.470804455448665, 3.961010168812714, 0.12048732474341543],
						[5.224378109973743, 2.502961393590481, 0.2984352779739605]]
	# count pareto-front
	function_values = np.array([list(map(f[i], X)) for i in range(f_num)])
	if np.shape(function_values)[-1] == 1:
		function_values = np.squeeze(function_values, -1)
	min_function_values = np.min(function_values, axis = 1)
	pareto_front = countHypervolume(function_values.T, np.array(min_function_values))
	if (function_type == "train_05" or function_type == "train_005" or
		function_type == "train" or function_type == "train_large" or
	 	function_type == "RBF_0.05" or function_type == "RBF_0.2" or 
		function_type == "matern52_0.05" or function_type == "matern52_0.2" or
		function_type == "RBF_0.05_large" or function_type == "matern52_0.05_large"):
		return f, pareto_front, min_function_values, kernels, kernel_lss
	else:
		return f, pareto_front, min_function_values, ["matern52"]*f_num, kernel_lss

def domain(function_type, domain_num, seed, d = 1, discrete = True):
	os.makedirs("domain", exist_ok=True)
	file_name = './domain/' + function_type + "_domain_" + str(domain_num) + ".npy"
	if discrete:
		if function_type == "NERF_synthetic" or function_type == "NERF_synthetic_fnum_3":
			scene, instance = get_NERF_scene()
			file_name = "./domain/" + "NERF_synthetic"  + "_" + scene + "_domain_" + str(domain_num) + ".npy"
		if os.path.exists(file_name):
			X = np.load(file_name)
			return X
	else:
		if (function_type == "train_05" or function_type == "train_005" or function_type == "train" or function_type == "train_large" or function_type == "RBF_0.05" or function_type == "RBF_0.2"  or function_type == "RBF_0.05_large" or function_type == "matern52_0.05" or function_type == "matern52_0.2" or function_type == "matern52_0.3"):
			X = sobol_seq.i4_sobol_generate(d, domain_num, seed)
		elif (function_type == "RE24" or function_type == "DR" or function_type == "BC" or function_type == "ARS" or function_type == "DRZ" or function_type == "Branin" or function_type == "Currin" or function_type == "AR" or function_type == "DR" or function_type == "ARa" or function_type == "BCD" or function_type == "ASR" or function_type == "ABC" or function_type == "ACD" or function_type == "ABD"):
			X = sobol_seq.i4_sobol_generate(2, domain_num, seed)
		elif (function_type == "RE25" or function_type == "RE22" or function_type == "RE31"):
			X = sobol_seq.i4_sobol_generate(3, domain_num, seed)
		elif (function_type == "RE33" or function_type == "RE21" or function_type == "RE23" or function_type == "RE32" or function_type == "RE36" or function_type == "RE37"):
			X = sobol_seq.i4_sobol_generate(4, domain_num, seed)
		elif (function_type == "RE34"):
			X = sobol_seq.i4_sobol_generate(5, domain_num, seed)
		elif (function_type == "RE35"):
			X = sobol_seq.i4_sobol_generate(7, domain_num, seed)
		elif (function_type == "NERF_real" or function_type == "NERF_synthetic" or function_type == "NERF_synthetic_fnum_3"):
			# clean data
			if function_type == "NERF_real":
				df = pandas.read_csv('./Environment/record_real.csv')
			else:
				scene, instance = get_NERF_scene()
				if scene != "chairs":
					df = pandas.read_csv('./Environment/record_synthetic_v2.csv')
				else:
					df = pandas.read_csv(f'./Environment/chairs/all/{instance}.csv')
			df = df.drop('dataset', axis = 1)
			df = df.drop('psnr', axis = 1)
			df = df.drop('ssim', axis = 1)
			df = df.drop('lpips', axis = 1)
			df = df.drop('npt', axis = 1)
			df = df.drop('size', axis = 1)

			scene, instance = get_NERF_scene()

			if scene != "chairs":
				df = df[df['scene'] == scene]

			df = df.drop('scene', axis = 1)
			# normalize data
			df=(df-df.min())/(df.max()-df.min())
			if scene == "chairs":
				df = df.fillna(0)
			X = df.values
			return X
		if not os.path.exists(file_name):
			np.save(file_name, X)
		return X

def countHypervolume(function_values, min_function_values):
	hv = get_performance_indicator("hv", ref_point=-1*min_function_values)
	return hv.do(-1*function_values)
