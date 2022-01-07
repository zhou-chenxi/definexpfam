#!/usr/bin/env python3

import os
import copy
from definexpfam.base_density import *
from definexpfam.basis_function import *
from definexpfam.negloglik_finexpfam import batch_montecarlo_params, negloglik_optalgoparams
from definexpfam.influence_function_contam_data_array import eval_IF_MLfindimexpfam_logdensity_contam_data_array

if __name__ == '__main__':
	
	os.chdir('/Users/chenxizhou/Dropbox/code_package/definexpfam')
	true_data = np.load('data/geyser.npy').astype(np.float64)
	df = copy.deepcopy(true_data[:, 0]).reshape(-1, 1)
	
	# original data with 108.0 removed
	data_waiting = df[df != 108.0]
	
	# array of contaminated data
	contam_data_array = np.arange(2., 410., 4)
	# np.sort(np.unique(np.concatenate([np.arange(2., 410., 4), np.arange(40., 100., 2)])))
	# np.sort(np.unique(np.concatenate((np.arange(90., 401., 2), data_waiting.flatten()))))
	
	# basis function used
	basisfunction_name = 'Gaussian'
	
	# bandwidth parameter in the Gaussian basis function
	bw = 9.0
	
	print(f'bw={bw}.')
	
	# contamnation weight
	contam_weight = 1e-2
	
	# base density
	base_density = BasedenGamma(np.load('data/geyser.npy').astype(np.float64)[:, 0])
	plot_xlimit = (1., 410.)
	plot_cnt = 3000
	new_data = np.linspace(plot_xlimit[0], plot_xlimit[1], plot_cnt)
	
	# grid points
	landmarks = np.arange(1., 411., 2).reshape(-1, 1)  # np.arange(42., 101., 2).reshape(-1, 1) # np.arange(1., 411., 2).reshape(-1, 1)
	# np.arange(42., 101., 2).reshape(-1, 1) # np.arange(1., 411., 2).reshape(-1, 1)
	landmarks_name = 'np.arange(1., 411., 2).reshape(-1, 1)'
	print(f'{len(landmarks)} basis functions are used.')
	
	# batch Monte Carlo parameters
	bmc_params = batch_montecarlo_params(
		mc_batch_size=5000,
		mc_tol=5e-3)
	
	# gradient descent algorithm parameters
	gdalgo_params = negloglik_optalgoparams(
		start_pt=np.zeros((landmarks.shape[0], 1)),
		step_size=0.2,
		max_iter=100,
		rel_tol=1e-5,
		abs_tol=0.05)
	abstol = gdalgo_params['abs_tol']
	stepsize = gdalgo_params['step_size']
	random_seed_nums = [0]
	
	print(f"Step size = {stepsize}.")
	print(f"Absolute tolerance parameter = {abstol}.")
	for i in random_seed_nums:
		
		print(f'Number of basis functions = {len(landmarks)}')
		np.random.seed(i)
		print(f'random number = {i}')
		
		result = eval_IF_MLfindimexpfam_logdensity_contam_data_array(
			data=data_waiting,
			new_data=new_data,
			contam_data_array=contam_data_array,
			contam_weight=contam_weight,
			landmarks=landmarks,
			bw=bw,
			degree=None,
			base_density=base_density,
			optalgo_params=gdalgo_params,
			batchmc_params=bmc_params,
			basisfunction_name=basisfunction_name,
			save_data=True,
			save_dir=f'FinExpFam-ML-basisn={len(landmarks)}-landmarks={landmarks_name}-bw={bw}-basisfunction={basisfunction_name}-contamweight={contam_weight}-plotdomain={plot_xlimit}-plotcnts={plot_cnt}-abstol={abstol}-stepsize={stepsize}-seed={i}',
			print_error=True)
