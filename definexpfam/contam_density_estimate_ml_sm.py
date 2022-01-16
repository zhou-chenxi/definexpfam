from scipy import integrate
from definexpfam.check import *
from definexpfam.basis_function import *
from definexpfam.base_density import *
from definexpfam.negloglik_finexpfam import negloglik_finexpfam_grad_logpar_batchmc


class ContamSMFinExpFam:
	
	"""
	A class of estimating the probability density function via minimizing the score matching loss function
	over a finite-dimensional exponential family in the presence of a contaminated observation.
	
	...
	
	Attributes
	----------
	data : numpy.ndarray
		The array of observations whose probability density function is to be estimated.
		
	contam_data : numpy.ndarray
		The array of contaminated observation.
		
	N : int
		The number of observations in self.data.
		
	n : int
		The number of observations in self.contam_data; must be equal to 1.
		
	d : int
		The dimensionality of self.data and self.contam_data; must be equal to 1.
		
	contam_weight : float
		The weight of contamination.
		
	base_density : base_density object
		The base density function used to estimate the probability density function.
	
	basisfunction_name : str
		The name of the basis functions; must be one of 'Polynomial', 'Gaussian', 'RationalQuadratic',
		'Logistic', 'Triweight', and 'Sigmoid'.
	
	basis_function : basis_function object
		The basis function used to estimate the probability density function.
		__type__ must be 'basis_function'.
		
	bw : float
		The bandwidth parameter in the basis function when self.basisfunction_name is one of 'Gaussian',
		'RationalQuadratic', 'Logistic', 'Triweight', and 'Sigmoid'.

	landmarks : numpy.ndarray or None
		The array at which the basis functions are centered;
		only works when basisfunction_name is one of 'Gaussian', 'RationalQuadratic', 'Logistic',
		'Triweight', and 'Sigmoid'.
	
	n_basis : int
		The number of basis functions.
	
	degree : int or None
		The degree of the polynomial when self.basisfunction_name is 'Polynomial'; default is None.
	
	Methods
	-------
	coef(pen_param=0.)
		Returns the natural parameter that minimizes the score matching loss function
		in a finite-dimensional exponential family.
	
	natparam(new_data, coef)
		Evaluates the inner product between the natural parameter and the sufficient statistic
		in the score matching density estimate at new_data.
	
	unnormalized_density_eval_1d(x, coef, minus_const=0)
		Evaluates the density function up to a normalizing constant at 1-dimensional data x.
		
	log_partition_1d(coef, minus_const=0.)
		Evaluates the log-partition function at coef.
	
	log_density(new_data, coef, minus_const=0, compute_base_density=False)
		Evaluates the log-density function at new_data.
	
	"""
	
	def __init__(self, data, contam_data, contam_weight, base_density,
				 landmarks=None, bw=None, degree=None, basisfunction_name='Gaussian'):
		
		"""
		Parameters
		----------
		data : numpy.ndarray
			The array of observations whose probability density function is to be estimated.

		contam_data : numpy.ndarray
			The array of contaminated observation.

		contam_weight : float
			The weight of contamination.
		
		base_density : base_density object
			The base density function used to estimate the probability density function.
		
		landmarks : numpy.ndarray or None, optional
			The array at which the basis functions are centered;
			only works when basisfunction_name is one of 'Gaussian', 'RationalQuadratic', 'Logistic',
			'Triweight', and 'Sigmoid'; default is None.
		
		bw : float or None, optional
			The bandwidth parameter when basisfunction_name is one of 'Gaussian', 'RationalQuadratic', 'Logistic',
			'Triweight', and 'Sigmoid'; default is None.
			If not None, must be strictly positive.
		
		degree : int or None, optional
			The degree of the polynomial when basisfunction_name is 'Polynomial'; default is None.
		
		basisfunction_name : str, optional
			The name of the basis function in the finite-dimensional exponential family;
			must be one of 'Polynomial', 'Gaussian', 'RationalQuadratic', 'Logistic', 'Triweight', and 'Sigmoid';
			default is 'Gaussian'.
		
		"""
		
		# check types of data and contam_data
		if isinstance(data, np.ndarray):
			data = np.array(data)
		
		if isinstance(contam_data, np.ndarray):
			contam_data = np.array(contam_data)
		
		if isinstance(landmarks, np.ndarray):
			landmarks = np.array(landmarks)
		
		# check compatibility of data and contam_data
		if len(data.shape) == 1:
			data = data.reshape(-1, 1)
		
		if len(contam_data.shape) == 1:
			contam_data = contam_data.reshape(-1, 1)
		
		if len(landmarks.shape) == 1:
			landmarks = landmarks.reshape(-1, 1)
		
		N, d = data.shape
		n, d1 = contam_data.shape
		m, d2 = landmarks.shape
		
		if d != d1 or d != d2 or d1 != d2:
			raise ValueError('The shape of data, contam_data and landmarks are not compatible.')
		
		if n != 1:
			raise ValueError('There are multiple contaminated data. Please just supply one.')
		
		self.data = data
		self.contam_data = contam_data
		self.N = N
		self.n = n
		self.d = d
		
		# check the validity of the contam_weight
		assert 0. <= contam_weight <= 1., 'contam_weight must be between 0 and 1, inclusively.'
		self.contam_weight = contam_weight
		
		# check the base density
		check_basedensity(base_density)
		self.base_density = base_density
		
		# check the validity of basisfunction_name
		if basisfunction_name not in ['Polynomial', 'Gaussian', 'RationalQuadratic',
									  'Logistic', 'Triweight', 'Sigmoid']:
			raise ValueError("basisfunction_name must be one of 'Polynomial', " +
							 "'Gaussian', 'RationalQuadratic', 'Logistic', 'Triweight', 'Sigmoid', " +
							 f'but got {basisfunction_name}.')
		
		# check the validity of banwidth parameter
		if basisfunction_name in ['Gaussian', 'RationalQuadratic', 'Logistic', 'Triweight', 'Sigmoid']:
			
			if bw is None:
				
				raise ValueError(f'The {basisfunction_name} basis function is used, and bw cannot be None.')
			
			elif bw <= 0.:
				
				raise ValueError('The bw parameter must be strictly positive.')
			
			self.basisfunction_name = basisfunction_name
			self.bw = bw
			self.landmarks = landmarks
			
			if self.basisfunction_name == 'Gaussian':
				
				self.basis_function = GaussianBasisFunction(
					landmarks=self.landmarks,
					bw=self.bw)
				self.n_basis = landmarks.shape[0]
			
			elif self.basisfunction_name == 'RationalQuadratic':
				
				self.basis_function = RationalQuadraticBasisFunction(
					landmarks=self.landmarks,
					bw=self.bw)
				self.n_basis = landmarks.shape[0]
			
			elif self.basisfunction_name == 'Logistic':
				
				self.basis_function = LogisticBasisFunction(
					landmarks=self.landmarks,
					bw=self.bw)
				self.n_basis = landmarks.shape[0]
			
			elif self.basisfunction_name == 'Triweight':
				
				self.basis_function = TriweightBasisFunction(
					landmarks=self.landmarks,
					bw=self.bw)
				self.n_basis = landmarks.shape[0]
			
			elif self.basisfunction_name == 'Sigmoid':
				
				self.basis_function = SigmoidBasisFunction(
					landmarks=self.landmarks,
					bw=self.bw)
				self.n_basis = landmarks.shape[0]
		
		# check the validity of degree
		elif basisfunction_name == 'Polynomial':
			
			if degree is None:
				
				raise ValueError('Polynomial basis function is used, and degree cannot be None.')
			
			elif not isinstance(degree, int):
				
				degree = int(degree)
				
				if degree <= 0:
					raise ValueError('degree cannot be negative.')
			
			self.basisfunction_name = basisfunction_name
			self.degree = degree
			
			self.basis_function = PolynomialBasisFunction(
				landmarks=self.landmarks,
				degree=self.degree)
			self.n_basis = self.degree
	
	def coef(self, pen_param=0.):
		
		"""
		Returns the natural parameter that minimizes the score matching loss function
		in a finite-dimensional exponential family.
		
		Parameters
		----------
		pen_param : float
			The penalty parameter on the size of the natural parameter to ensure the numerical stability;
			must be non-negative.
			
		Returns
		-------
		numpy.ndarray
			An array of the natural parameter in the score matching density estimate.
		
		"""
		
		N, d = self.N, self.d
		if self.basisfunction_name == 'Polynomial':
			
			n_basis = self.basis_function.degree
		
		else:
			
			n_basis = self.basis_function.landmarks.shape[0]
		
		# compute sum_{i=1}^N (DT(X_i) DT(X_i)^\top) / N
		DT_data = self.basis_function.basisfunction_deriv1(self.data)
		dt_prod_term_data = sum([np.matmul(DT_data[:, i].reshape(n_basis, d),
										   DT_data[:, i].reshape(n_basis, d).T) for i in range(N)]) / N
		
		# compute (DT(y) DT(y)^\top)
		DT_contamdata = self.basis_function.basisfunction_deriv1(self.contam_data)
		dt_prod_term_contamdata = np.matmul(DT_contamdata[:, 0].reshape(n_basis, d),
											DT_contamdata[:, 0].reshape(n_basis, d).T)
		
		LHS = (1. - self.contam_weight) * dt_prod_term_data + self.contam_weight * dt_prod_term_contamdata
		
		# compute the matrix G at data, which involves the second derivative
		matG_data = self.basis_function.basisfunction_deriv2(self.data)
		sum_matG_data = np.sum(sum([matG_data[:, i].reshape(n_basis, d) for i in range(N)]),
							   axis=1, keepdims=True) / N
		
		# compute the matrix G at contam_data, which involves the second derivative
		matG_contamdata = self.basis_function.basisfunction_deriv2(self.contam_data)
		sum_matG_contamdata = np.sum(matG_contamdata[:, 0].reshape(n_basis, d),
									 axis=1, keepdims=True)
		
		# compute DT and grad of log mu
		# compute the gradient of log mu at data
		# each row corresponds to one data point
		grad_logmu_data = np.array([self.base_density.logbaseden_deriv1(self.data, j).flatten() for j in range(d)]).T
		dt_baseden_term_data = sum([np.matmul(DT_data[:, i].reshape(n_basis, d),
											  grad_logmu_data[[i]].T) for i in range(N)]) / N
		
		grad_logmu_contamdata = np.array \
			([self.base_density.logbaseden_deriv1(self.contam_data, j).flatten() for j in range(d)]).T
		dt_baseden_term_contamdata = np.matmul(DT_contamdata[:, 0].reshape(n_basis, d),
											   grad_logmu_contamdata[[0]].T)
		
		RHS = ((1. - self.contam_weight) * (dt_baseden_term_data + sum_matG_data) +
			   self.contam_weight * (dt_baseden_term_contamdata + sum_matG_contamdata))
		
		coef = -np.linalg.lstsq(LHS + pen_param * np.eye(n_basis), RHS, rcond=None)[0]
		
		return coef
	
	def natparam(self, new_data, coef):
		
		"""
		Evaluates the inner product between the natural parameter and the sufficient statistic
		in the score matching density estimate at new_data.
		
		Parameters
		----------
		new_data : numpy.ndarray
			The array of data at which the natural parameter is to be evaluated.
			
		coef : numpy.ndarray
			The value of the natural parameter.
		
		Returns
		-------
		numpy.ndarray
			The 1-dimensional array of the values of the inner product
			between the natural parameter and the sufficient statistic  at new_data.
		
		"""
		
		output = np.matmul(self.basis_function.basisfunction_eval(new_data).T, coef).flatten()
		
		return output
	
	def unnormalized_density_eval_1d(self, x, coef, minus_const=0):
		
		"""
		Evaluates the density function up to a normalizing constant at 1-dimensional data x.
		This function is mainly used in computing the normalizing constant and only works when self.d is equal to 1.
		
		Parameters
		----------
		x : float or numpy.ndarray
			The point at which the un-normalized density function is to be evaluated.
			
		coef : numpy.ndarray
			The value of the natural parameter.
		
		minus_const : float
			A constant to be subtracted in the exponent to ensure the finite-ness of numerical integration.
		
		Returns
		-------
		float or numpy.ndarray
			The value of the un-normalized density function at x.
		
		"""
		
		if self.basis_function.basisfunction_name == 'Polynomial':
			
			den = (self.base_density.baseden_eval_1d(x) *
				   np.exp(- minus_const + np.sum([coef[i] * x ** (i + 1) for i in range(self.basis_function.degree)])))
		
		elif self.basis_function.basisfunction_name in ['Gaussian', 'RationalQuadratic', 'Logistic', 'Triweight',
														'Sigmoid']:
			
			n_basis = self.basis_function.landmarks.shape[0]
			
			den = (self.base_density.baseden_eval_1d(x) *
				   np.exp(- minus_const + np.sum([coef[i] * self.basis_function.basis_x_1d(self.landmarks[i].item())(x)
												  for i in range(n_basis)])))
		return den
	
	def log_partition_1d(self, coef, minus_const=0.):
		
		"""
		Evaluates the log-partition function at coef.
		
		Parameters
		----------
		coef : numpy.ndarray
			The value of the natural parameter.
		
		minus_const : float
			A constant to be subtracted in the exponent to ensure the finite-ness of numerical integration.
		
		Returns
		-------
		float
			The value of the log-partition function at coef.

		"""
		
		if self.d != 1:
			error_msg = (f'The function self.log_partition_1d only works for 1-dimensional data. '
						 f'But the underlying data is {self.d}-dimensional.')
			raise ValueError(error_msg)
		
		norm_const, _ = integrate.quad(
			func=self.unnormalized_density_eval_1d,
			a=self.base_density.domain[0][0],
			b=self.base_density.domain[0][1],
			args=(coef, minus_const,),
			limit=100)
		
		output = np.log(norm_const)
		
		return output
	
	def log_density(self, new_data, coef, minus_const=0, compute_base_density=False):
		
		"""
		Evaluates the log-density function at new_data.
		
		Parameters
		----------
		new_data : numpy.ndarray
			The array of data at which the log-density function is to be evaluated.
		
		coef : numpy.ndarray
			The value of the natural parameter.
			
		minus_const : float
			A constant to be subtracted in the exponent to ensure the finite-ness of numerical integration.
		
		compute_base_density : bool, optional
			Whether to compute the base density part; default is False.
		
		Returns
		-------
		numpy.ndarray
			An 1-dimensional array of the values of the log-density function at new_data.
			
		"""
		
		if compute_base_density:
			
			baseden_part = np.log(self.base_density.baseden_eval(new_data).flatten())
		
		else:
			
			baseden_part = 0.
		
		logpar = self.log_partition_1d(coef=coef, minus_const=minus_const)
		natparam_part = self.natparam(
			new_data=new_data,
			coef=coef)
		
		print(logpar)
		final_result = baseden_part + natparam_part - logpar - minus_const
		
		return final_result


class ContamMLFinExpFam:
	
	"""
	A class of estimating the probability density function via minimizing the negative log-likelihood loss function
	over a finite-dimensional exponential family in the presence of a contaminated observation.
	
	...
	
	Attributes
	----------
	data : numpy.ndarray
		The array of observations whose probability density function is to be estimated.
		
	contam_data : numpy.ndarray
		The array of contaminated observation.
		
	N : int
		The number of observations in self.data.
		
	n : int
		The number of observations in self.contam_data; must be equal to 1.
		
	d : int
		The dimensionality of self.data and self.contam_data; must be equal to 1.
		
	contam_weight : float
		The weight of contamination.
		
	base_density : base_density object
		The base density function used to estimate the probability density function.
	
	basisfunction_name : str
		The name of the basis functions; must be one of 'Polynomial', 'Gaussian', 'RationalQuadratic',
		'Logistic', 'Triweight', and 'Sigmoid'.
	
	basis_function : basis_function object
		The basis function used to estimate the probability density function.
		__type__ must be 'basis_function'.
		
	bw : float
		The bandwidth parameter in the basis function when self.basisfunction_name is one of 'Gaussian',
		'RationalQuadratic', 'Logistic', 'Triweight', and 'Sigmoid'.

	landmarks : numpy.ndarray or None
		The array at which the basis functions are centered;
		only works when basisfunction_name is one of 'Gaussian', 'RationalQuadratic', 'Logistic',
		'Triweight', and 'Sigmoid'.
	
	n_basis : int
		The number of basis functions.
	
	degree : int or None
		The degree of the polynomial when self.basisfunction_name is 'Polynomial'; default is None.

	Methods
	-------
	coef(optalgo_params, batchmc_params, print_error=True)
		Returns the natural parameter that minimizes the negative log-likelihood loss function
		in a finite-dimensional exponential family.
	
	natparam(new_data, coef)
		Evaluates the inner product between the natural parameter and the sufficient statistic
		in the maximum likelihood density estimate at new_data.
	
	unnormalized_density_eval_1d(x, coef, minus_const=0)
		Evaluates the density function up to a normalizing constant at 1-dimensional data x.
	
	log_partition_1d(coef, minus_const=0)
		Evaluates the log-partition function at coef.
	
	log_density(new_data, coef, minus_const=0, compute_base_density=False)
		Evaluates the log-density function at new_data.
		
	"""
	
	def __init__(self, data, contam_data, contam_weight, base_density,
				 landmarks=None, bw=None, degree=None, basisfunction_name='Gaussian'):
		
		"""
		Parameters
		----------
		data : numpy.ndarray
			The array of observations whose probability density function is to be estimated.

		contam_data : numpy.ndarray
			The array of contaminated observation.

		contam_weight : float
			The weight of contamination.
		
		base_density : base_density object
			The base density function used to estimate the probability density function.
		
		landmarks : numpy.ndarray or None, optional
			The array at which the basis functions are centered;
			only works when basisfunction_name is one of 'Gaussian', 'RationalQuadratic', 'Logistic',
			'Triweight', and 'Sigmoid';
			default is None.
		
		bw : float or None, optional
			The bandwidth parameter when basisfunction_name is one of 'Gaussian', 'RationalQuadratic', 'Logistic',
			'Triweight', and 'Sigmoid'; default is None.
			If not None, must be strictly positive.
		
		degree : int or None, optional
			The degree of the polynomial when basisfunction_name is 'Polynomial'; default is None.
		
		basisfunction_name : str, optional
			The name of the basis function in the finite-dimensional exponential family;
			must be one of 'Polynomial', 'Gaussian', 'RationalQuadratic', 'Logistic', 'Triweight', and 'Sigmoid';
			default is 'Gaussian'.
			
		"""
		
		# check types of data and contam_data
		if isinstance(data, np.ndarray):
			data = np.array(data)
		
		if isinstance(contam_data, np.ndarray):
			contam_data = np.array(contam_data)
		
		if isinstance(landmarks, np.ndarray):
			landmarks = np.array(landmarks)
		
		# check compatibility of data and contam_data
		if len(data.shape) == 1:
			data = data.reshape(-1, 1)
		
		if len(contam_data.shape) == 1:
			contam_data = contam_data.reshape(-1, 1)
		
		if len(landmarks.shape) == 1:
			landmarks = landmarks.reshape(-1, 1)
		
		N, d = data.shape
		n, d1 = contam_data.shape
		m, d2 = landmarks.shape
		
		if d != d1 or d != d2 or d1 != d2:
			raise ValueError('The shape of data, contam_data and landmarks are not compatible.')
		
		if n != 1:
			raise ValueError('There are multiple contaminated data. Please just supply one.')
		
		self.data = data
		self.contam_data = contam_data
		self.N = N
		self.n = n
		self.d = d
		
		# check the validity of the contam_weight
		assert 0. <= contam_weight <= 1., 'contam_weight must be between 0 and 1, inclusively.'
		self.contam_weight = contam_weight
		
		# check the base density
		check_basedensity(base_density)
		self.base_density = base_density
		
		# check the validity of basisfunction_name
		if basisfunction_name not in ['Polynomial', 'Gaussian', 'RationalQuadratic',
									  'Logistic', 'Triweight', 'Sigmoid']:
			raise ValueError("basisfunction_name must be one of 'Polynomial', " +
							 "'Gaussian', 'RationalQuadratic', 'Logistic', 'Triweight', 'Sigmoid', " +
							 f'but got {basisfunction_name}.')
		
		# check the validity of banwidth parameter
		if basisfunction_name in ['Gaussian', 'RationalQuadratic', 'Logistic', 'Triweight', 'Sigmoid']:
			
			if bw is None:
				
				raise ValueError(f'The {basisfunction_name} basis function is used, and bw cannot be None.')
			
			elif bw <= 0.:
				
				raise ValueError('The bw parameter must be strictly positive.')
			
			self.basisfunction_name = basisfunction_name
			self.bw = bw
			self.landmarks = landmarks
			
			if self.basisfunction_name == 'Gaussian':
				
				self.basis_function = GaussianBasisFunction(
					landmarks=self.landmarks,
					bw=self.bw)
			
			elif self.basisfunction_name == 'RationalQuadratic':
				
				self.basis_function = RationalQuadraticBasisFunction(
					landmarks=self.landmarks,
					bw=self.bw)
			
			elif self.basisfunction_name == 'Logistic':
				
				self.basis_function = LogisticBasisFunction(
					landmarks=self.landmarks,
					bw=self.bw)
			
			elif self.basisfunction_name == 'Triweight':
				
				self.basis_function = TriweightBasisFunction(
					landmarks=self.landmarks,
					bw=self.bw)
			
			elif self.basisfunction_name == 'Sigmoid':
				
				self.basis_function = SigmoidBasisFunction(
					landmarks=self.landmarks,
					bw=self.bw)
		
		# check the validity of degree
		elif basisfunction_name == 'Polynomial':
			
			if degree is None:
				
				raise ValueError('Polynomial basis function is used, and degree cannot be None.')
			
			elif not isinstance(degree, int):
				
				degree = int(degree)
				
				if degree <= 0:
					raise ValueError('degree cannot be negative.')
			
			self.basisfunction_name = basisfunction_name
			self.degree = degree
			
			self.basis_function = PolynomialBasisFunction(
				landmarks=self.landmarks,
				degree=self.degree)
	
	def coef(self, optalgo_params, batchmc_params, print_error=True):
		
		"""
		Returns the natural parameter that minimizes the negative log-likelihood loss function
		in a finite-dimensional exponential family.
		
		Parameters
		----------
		optalgo_params : dict
			The dictionary of parameters to control the gradient descent algorithm.
			Must be returned from the function negloglik_optalgo_params.
		
		batchmc_params : dict
			The dictionary of parameters to control the batch Monte Carlo method
			to approximate the partition function and the gradient of the log-partition function.
			Must be returned from the function batch_montecarlo_params.
		
		print_error : bool, optional
			Whether to print the error of the gradient descent algorithm at each iteration; default is True.
		
		Returns
		-------
		numpy.ndarray
			An array of the natural parameter in the maximum likelihood density estimate.
		
		"""
		
		N, d = self.N, self.d
		
		# parameters associated with gradient descent algorithm
		start_pt = optalgo_params["start_pt"]
		step_size = optalgo_params["step_size"]
		max_iter = optalgo_params["max_iter"]
		rel_tol = optalgo_params["rel_tol"]
		abs_tol = optalgo_params["abs_tol"]
		
		if not isinstance(step_size, float):
			raise TypeError(("The type of step_size in optalgo_params should be float, but got {}".format(
				type(step_size))))
		
		if step_size <= 0.:
			raise ValueError("The step_size in optalgo_params must be strictly positive, but got {}.".format(
				step_size))
		
		# parameters associated with batch Monte Carlo estimation
		mc_batch_size = batchmc_params["mc_batch_size"]
		mc_tol = batchmc_params["mc_tol"]
		
		# the gradient of the loss function is
		# nabla L (alpha) = nabla A (alpha) - (1 / n) gram_matrix boldone_n
		# the gradient descent update is
		# new_iter = current_iter - step_size * nabla L (alpha)
		
		current_iter = start_pt.reshape(-1, 1)
		
		# compute the gradient of the log-partition function at current_iter
		mc_output1, mc_output2 = negloglik_finexpfam_grad_logpar_batchmc(
			data=self.data,
			basis_function=self.basis_function,
			base_density=self.base_density,
			coef=current_iter,
			batch_size=mc_batch_size,
			tol_param=mc_tol,
			compute_grad=True,
			print_error=False)
		
		grad_logpar = mc_output2.reshape(-1, 1)
		
		# form the Gram matrix
		gram_data = self.basis_function.basisfunction_eval(self.data)
		gram_contamdata = self.basis_function.basisfunction_eval(self.contam_data)
		
		grad_term2 = ((1 - self.contam_weight) * gram_data.mean(axis=1, keepdims=True) +
					  self.contam_weight * gram_contamdata)
		
		# compute the gradient of the loss function at current_iter
		current_grad = grad_logpar - grad_term2
		
		# compute the updated iter
		new_iter = current_iter - step_size * current_grad
		
		# compute the error of the first update
		grad0_norm = np.linalg.norm(current_grad, 2)
		grad_new_norm = grad0_norm
		error = grad0_norm / (grad0_norm + 1e-8)
		# np.linalg.norm(new_iter - current_iter, 2) / (np.linalg.norm(current_iter, 2) + 1e-1)
		
		iter_num = 1
		
		if print_error:
			print("Iter = {iter_num}, GradNorm = {gradnorm}, Relative Error = {error}".format(
				iter_num=iter_num, gradnorm=grad0_norm, error=error))
		
		while error > rel_tol and grad_new_norm > abs_tol and iter_num < max_iter:
			
			current_iter = new_iter
			
			# compute the gradient at current_iter
			mc_output1, mc_output2 = negloglik_finexpfam_grad_logpar_batchmc(
				data=self.data,
				basis_function=self.basis_function,
				base_density=self.base_density,
				coef=current_iter,
				batch_size=mc_batch_size,
				tol_param=mc_tol,
				compute_grad=True,
				print_error=False)
			
			grad_logpar = mc_output2.reshape(-1, 1)
			
			# compute the gradient of the loss function
			current_grad = grad_logpar - grad_term2
			
			# compute the updated iter
			new_iter = current_iter - step_size * current_grad
			
			# compute the error of the first update
			grad_new_norm = np.linalg.norm(current_grad, 2)
			error = grad_new_norm / grad0_norm
			# np.linalg.norm(new_iter - current_iter, 2) / (np.linalg.norm(current_iter, 2) + 1e-1)
			
			iter_num += 1
			
			if print_error:
				print("Iter = {iter_num}, GradNorm = {gradnorm}, Relative Error = {error}".format(
					iter_num=iter_num, gradnorm=grad_new_norm, error=error))
		
		coefficients = new_iter
		
		return coefficients
	
	def natparam(self, new_data, coef):
		
		"""
		Evaluates the inner product between the natural parameter and the sufficient statistic
		in the maximum likelihood density estimate at new_data.
		
		Parameters
		----------
		new_data : numpy.ndarray
			The array of data at which the natural parameter is to be evaluated.
			
		coef : numpy.ndarray
			The value of the natural parameter.
		
		Returns
		-------
		numpy.ndarray
			The 1-dimensional array of the values of the inner product
			between the natural parameter and the sufficient statistic  at new_data.
		
		"""
		
		output = np.matmul(self.basis_function.basisfunction_eval(new_data).T, coef).flatten()
		
		return output
	
	def unnormalized_density_eval_1d(self, x, coef, minus_const=0):
		
		"""
		Evaluates the density function up to a normalizing constant at 1-dimensional data x.
		This function is mainly used in computing the normalizing constant and only works when self.d is equal to 1.
		
		Parameters
		----------
		x : float or numpy.ndarray
			The point at which the un-normalized density function is to be evaluated.
			
		coef : numpy.ndarray
			The value of the natural parameter.
		
		minus_const : float
			A constant to be subtracted in the exponent to ensure the finite-ness of numerical integration.
		
		Returns
		-------
		float or numpy.ndarray
			The value of the un-normalized density function at x.
		
		"""
		
		if self.basis_function.basisfunction_name == 'Polynomial':
			
			den = (self.base_density.baseden_eval_1d(x) *
				   np.exp(-minus_const + np.sum([coef[i] * x ** (i + 1) for i in range(self.basis_function.degree)])))
		
		elif self.basis_function.basisfunction_name in ['Gaussian', 'RationalQuadratic', 'Logistic', 'Triweight',
														'Sigmoid']:
			
			n_basis = self.basis_function.landmarks.shape[0]
			
			den = (self.base_density.baseden_eval_1d(x) *
				   np.exp(-minus_const + np.sum([coef[i] * self.basis_function.basis_x_1d(self.landmarks[i].item())(x)
												 for i in range(n_basis)])))
		return den
	
	def log_partition_1d(self, coef, minus_const=0):
		
		"""
		Evaluates the log-partition function at coef.
		
		Parameters
		----------
		coef : numpy.ndarray
			The value of the natural parameter.
		
		minus_const : float
			A constant to be subtracted in the exponent to ensure the finite-ness of numerical integration.
		
		Returns
		-------
		float
			The value of the log-partition function at coef.

		"""
		
		
		if self.d != 1:
			error_msg = (f'The function self.log_partition_1d only works for 1-dimensional data. '
						 f'But the underlying data is {self.d}-dimensional.')
			raise ValueError(error_msg)
		
		norm_const, _ = integrate.quad(
			func=self.unnormalized_density_eval_1d,
			a=self.base_density.domain[0][0],
			b=self.base_density.domain[0][1],
			args=(coef, minus_const,),
			limit=100)
		
		output = np.log(norm_const)
		
		return output
	
	def log_density(self, new_data, coef, minus_const=0, compute_base_density=False):
		
		"""
		Evaluates the log-density function at new_data.
		
		Parameters
		----------
		new_data : numpy.ndarray
			The array of data at which the log-density function is to be evaluated.
		
		coef : numpy.ndarray
			The value of the natural parameter.
			
		minus_const : float
			A constant to be subtracted in the exponent to ensure the finite-ness of numerical integration.
		
		compute_base_density : bool, optional
			Whether to compute the base density part; default is False.
		
		Returns
		-------
		numpy.ndarray
			An 1-dimensional array of the values of the log-density function at new_data.
			
		"""
		
		if compute_base_density:
			
			baseden_part = np.log(self.base_density.baseden_eval(new_data).flatten())
		
		else:
			
			baseden_part = 0.
		
		logpar = self.log_partition_1d(coef=coef, minus_const=minus_const)
		natparam_part = self.natparam(
			new_data=new_data,
			coef=coef)
		
		print(logpar)
		final_result = baseden_part + natparam_part - logpar - minus_const
		
		return final_result
