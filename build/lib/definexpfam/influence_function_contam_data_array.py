import os
import numpy as np
from definexpfam.contam_density_estimate_ml_sm import ContamMLFinExpFam, ContamSMFinExpFam


def eval_IF_MLfindimexpfam_logdensity_contam_data_array(
		data, new_data, contam_data_array, contam_weight, landmarks, bw, degree,
		base_density, optalgo_params, batchmc_params, basisfunction_name='Gaussian',
		save_data=False, save_dir=None, print_error=True):
	
	"""

	Parameters
	----------


	Returns
	-------


	"""
	
	if contam_weight == 0.:
		raise ValueError('In order to compute the influence function, contam_weight cannot be 0.')
	
	# check the validity of the contam_data_array
	if not isinstance(contam_data_array, np.ndarray):
		raise TypeError(f'contam_data_array must be a numpy.ndarray, but got {type(contam_data_array)}.')
	
	# check the compatibility of data and new_data
	if not isinstance(data, np.ndarray):
		data = np.array(data)
	
	if not isinstance(new_data, np.ndarray):
		new_data = np.array(new_data)
	
	if not isinstance(landmarks, np.ndarray):
		landmarks = np.array(landmarks)
	
	if len(data.shape) == 1:
		data = data.reshape(-1, 1)
	
	if len(new_data.shape) == 1:
		new_data = new_data.reshape(-1, 1)
	
	if len(landmarks.shape) == 1:
		landmarks = landmarks.reshape(-1, 1)
	
	N, d = data.shape
	n, d1 = new_data.shape
	m, d2 = landmarks.shape
	if d != d1 or d != d2:
		raise ValueError('The shapes of data, new_data and landmarks are not compatible.')
	
	# check the compatibility of data and contam_data_array
	if len(contam_data_array.shape) == 1:
		contam_data_array = contam_data_array.reshape(-1, 1)
	if contam_data_array.shape[1] != d:
		raise ValueError('contam_data_array are not compatible with data and new_data.')
	
	print('-' * 50)
	print('Computing the uncontaminated log-density values.')
	# compute the log-density values of the uncontaminated data
	uncontam_den = ContamMLFinExpFam(
		data=data,
		contam_data=contam_data_array[0].reshape(1, d),
		contam_weight=0.,
		base_density=base_density,
		landmarks=landmarks,
		bw=bw,
		degree=degree,
		basisfunction_name=basisfunction_name)
	
	uncontam_coef = uncontam_den.coef(
		optalgo_params=optalgo_params,
		batchmc_params=batchmc_params,
		print_error=print_error)
	
	uncontam_logdenvals_new = uncontam_den.log_density(
		new_data=new_data,
		coef=uncontam_coef,
		minus_const=0.,
		compute_base_density=False)
	
	# save data
	if save_data:
		full_save_folder = 'data/' + save_dir
		if not os.path.isdir(full_save_folder):
			os.mkdir(full_save_folder)
		
		file_name_newdata = f'/new_data.npy'
		np.save(full_save_folder + file_name_newdata, new_data)
		
		print('new_data saved.')
		
		file_name_grid_points = f'/landmarks.npy'
		np.save(full_save_folder + file_name_grid_points, landmarks)
		
		print('grid_points saved.')
		
		file_name_contamdata = f'/contam_data.npy'
		np.save(full_save_folder + file_name_contamdata, contam_data_array)
		
		print('contam_data_array saved.')
		
		file_name_coef = f'/uncontam-coef.npy'
		np.save(full_save_folder + file_name_coef, uncontam_coef)
		
		print('uncontam_coef saved.')
		
		file_name_diff = f'/uncontam-logden-newdata.npy'
		np.save(full_save_folder + file_name_diff, uncontam_logdenvals_new)
		print('uncontam_logdensity saved.')
	
	IF_output_new = {}
	IF_output_new['new_data'] = new_data
	
	for i in range(len(contam_data_array)):
		
		print('-' * 50)
		print(f'Computing the contaminated log-density values ')
		print(f'with the current contaminated data point being {contam_data_array[i]}.')
		
		contam_den = ContamMLFinExpFam(
			data=data,
			contam_data=contam_data_array[i].reshape(1, d),
			contam_weight=contam_weight,
			base_density=base_density,
			landmarks=landmarks,
			bw=bw,
			degree=degree,
			basisfunction_name=basisfunction_name)
		
		contam_coef = contam_den.coef(
			optalgo_params=optalgo_params,
			batchmc_params=batchmc_params,
			print_error=print_error)
		
		contam_logdenvals_new = contam_den.log_density(
			new_data=new_data,
			coef=contam_coef,
			minus_const=0.,
			compute_base_density=False)
		
		IF_newdata = (contam_logdenvals_new - uncontam_logdenvals_new) / contam_weight
		IF_output_new['contam ' + str(contam_data_array[i])] = IF_newdata
		
		if save_data:
			# save the coef
			IF_file_name_coef = f'/contam_data={contam_data_array[i]}-contam-coef.npy'
			np.save(full_save_folder + IF_file_name_coef, contam_coef)
			
			# save the log density values
			file_name_diff = f'/contam_data={contam_data_array[i]}-contam-logden-newdata.npy'
			np.save(full_save_folder + file_name_diff, contam_logdenvals_new)
			
			# save the IF
			IF_file_name_new = f'/contam_data={contam_data_array[i]}-IF-logden-newdata.npy'
			np.save(full_save_folder + IF_file_name_new, IF_newdata)
	
	return IF_output_new
