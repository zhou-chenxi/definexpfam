from definexpfam.basis_function import *
from definexpfam.base_density import *


def negloglik_finexpfam_grad_logpar_batchmc(data, basis_function, base_density, coef, batch_size, tol_param,
                                            compute_grad=True, print_error=False):
    
    """
    Approximates the partition function and the gradient of the log-partition function at coef.
    The approximation method used is the batch Monte Carlo method. Terminate the sampling process
    until the relative difference of two consecutive approximations is less than tol_param.

    Let Y_1, ..., Y_M be random samples from the base density.
    The partition function evaluated at coef is approximated by
    (1 / M) sum_{i=1}^M exp ( sum_{j=1}^m coef[j] T_j (Y_i) ),
    and the gradient of the log-partition function evaluated at coef is approximated by
    (1 / M) sum_{i=1}^M T_l (Y_i) exp ( sum_{j=1}^m coef[j] T_j (Y_i) - A(coef)), for all l = 1, ..., m,
    where A(coef) is the log-parition function at coef, and m is the number of basis functions.
    
    Parameters
    ----------
    data : numpy.ndarray
        The array of observations whose probability density function is to be estimated.
    
    basis_function : basis_function object
        The basis function used to estimate the probability density function.
        __type__ must be 'basis_function'.
        
    base_density : base_density object
        The base density function used to estimate the probability density function.
        __type__ must be 'base_density'.
        
    coef : numpy.ndarray
        The array of natural parameter at which the partition function
        and the gradient of the log-partition function are to be approximated.

    batch_size : int
        The batch size in the batch Monte Carlo method.

    tol_param : float
        The floating point number below which sampling in the batch Monte Carlo is terminated.
        The smaller the tol_param is, the more accurate the approximations are.
    
    compute_grad : bool, optional
        Whether to approximate the gradient of the log-partition function; default is True.

    print_error : bool, optional
        Whether to print the error in the batch Monte Carlo method; default is False.

    Returns
    -------
    float
        The approximation of the partition function evaluated at coef.
    
    numpy.ndarray
        The approximation of the gradient of the log-partition function evaluated at coef;
        only returns when compute_grad is True.
        
    """
    
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    
    if len(coef.shape) == 1:
        coef = coef.reshape(-1, 1)
    
    # N, d = data.shape
    if basis_function.basisfunction_name == 'Polynomial': 
        N = basis_function.degree
    else: 
        N = basis_function.landmarks.shape[0]

    ###########################################################################
    # estimate the normalizing constant 
    # first drawing
    mc_samples1 = base_density.sample(batch_size)
    mc_kernel_matrix1 = basis_function.basisfunction_eval(mc_samples1)
    unnorm_density_part1 = np.exp(np.matmul(mc_kernel_matrix1.T, coef))
    norm_const1 = np.mean(unnorm_density_part1)
    
    # second drawing 
    mc_samples2 = base_density.sample(batch_size)
    mc_kernel_matrix2 = basis_function.basisfunction_eval(mc_samples2)
    unnorm_density_part2 = np.exp(np.matmul(mc_kernel_matrix2.T, coef))
    norm_const2 = np.mean(unnorm_density_part2)
    
    norm_est_old = norm_const1
    norm_est_new = (norm_const1 + norm_const2) / 2
    
    error_norm = np.abs(norm_est_old - norm_est_new) / norm_est_old
    
    if print_error: 
        print('normalizing constant error = {error:.7f}'.format(error=error_norm))
    
    batch_cnt = 2
    
    while error_norm > tol_param:
        
        norm_est_old = norm_est_new
        
        # another draw
        mc_samples = base_density.sample(batch_size)
        mc_kernel_matrix = basis_function.basisfunction_eval(mc_samples)
        unnorm_density_part = np.exp(np.matmul(mc_kernel_matrix.T, coef))
        norm_const2 = np.mean(unnorm_density_part)
        
        # update the Monte Carlo estimation 
        norm_est_new = (norm_est_old * batch_cnt + norm_const2) / (batch_cnt + 1)

        batch_cnt += 1
        
        error_norm = np.abs(norm_est_old - norm_est_new) / norm_est_old
        
        if print_error: 
            print('normalizing constant error = {error:.7f}'.format(error=error_norm))
    
    normalizing_const = norm_est_new
    
    if compute_grad:
        
        if print_error:
            print("#" * 45 + "\nEstimating the gradient of the log-partition now.")
        
        mc_samples1 = base_density.sample(batch_size)
        mc_kernel_matrix1 = basis_function.basisfunction_eval(mc_samples1)
        density_part1 = np.exp(np.matmul(mc_kernel_matrix1.T, coef).flatten()) / normalizing_const
        exp_est1 = np.array([np.mean(mc_kernel_matrix1[l1, :] * density_part1)
                             for l1 in range(N)]).astype(np.float64).reshape(1, -1)[0]
        
        mc_samples2 = base_density.sample(batch_size)
        mc_kernel_matrix2 = basis_function.basisfunction_eval(mc_samples2)
        density_part2 = np.exp(np.matmul(mc_kernel_matrix2.T, coef).flatten()) / normalizing_const
        exp_est2 = np.array([np.mean(mc_kernel_matrix2[l1, :] * density_part2)
                             for l1 in range(N)]).astype(np.float64).reshape(1, -1)[0]
        
        grad_est_old = exp_est1
        grad_est_new = (exp_est1 + exp_est2) / 2
        
        error_grad = np.linalg.norm(grad_est_old - grad_est_new, 2) / (np.linalg.norm(grad_est_old, 2) * N)
        
        if print_error: 
            print('gradient error = {error:.7f}'.format(error=error_grad))
        
        batch_cnt = 2
        
        while error_grad > tol_param:
        
            grad_est_old = grad_est_new

            # another draw
            mc_samples = base_density.sample(batch_size)
            mc_kernel_matrix = basis_function.basisfunction_eval(mc_samples)
            density_part = np.exp(np.matmul(mc_kernel_matrix.T, coef).flatten()) / normalizing_const
            exp_est2 = np.array([np.mean(mc_kernel_matrix[l1, :] * density_part)
                                 for l1 in range(N)]).astype(np.float64).reshape(1, -1)[0]
            
            grad_est_new = (grad_est_old * batch_cnt + exp_est2) / (batch_cnt + 1)

            batch_cnt += 1

            error_grad = np.linalg.norm(grad_est_old - grad_est_new, 2) / (np.linalg.norm(grad_est_old, 2) * N)

            if print_error: 
                print('gradient error = {error:.7f}'.format(error=error_grad))
    
    if not compute_grad:
        return norm_est_new
    else: 
        return norm_est_new, grad_est_new


def negloglik_finexpfam_grad_logpar_batchmc_se(data, basis_function, base_density, coef, batch_size, tol_param,
                                               compute_grad=True, print_error=False):
    
    """
    Approximates the partition function and the gradient of the log-partition function at coef.
    The approximation method used is the batch Monte Carlo method. Terminate the sampling process
    until the standard deviation of the approximations is less than tol_param.
    
    Let Y_1, ..., Y_M be random samples from the base density.
    The partition function evaluated at coef is approximated by
    (1 / M) sum_{i=1}^M exp ( sum_{j=1}^m coef[j] T_j (Y_i) ),
    and the gradient of the log-partition function evaluated at coef is approximated by
    (1 / M) sum_{i=1}^M T_l (Y_i) exp ( sum_{j=1}^m coef[j] T_j (Y_i) - A(coef)), for all l = 1, ..., m,
    where A(coef) is the log-partition function, and m is the number of basis functions.
    
    Parameters
    ----------
    data : numpy.ndarray
        The array of observations whose probability density function is to be estimated.
    
    basis_function : basis_function object
        The basis function used to estimate the probability density function.
        __type__ must be 'basis_function'.
        
    base_density : base_density object
        The base density function used to estimate the probability density function.
        __type__ must be 'base_density'.
        
    coef : numpy.ndarray
        The array of natural parameter at which the partition function
        and the gradient of the log-partition function are to be approximated.

    batch_size : int
        The batch size in the batch Monte Carlo method.

    tol_param : float
        The floating point number below which sampling in the batch Monte Carlo is terminated.
        The smaller the tol_param is, the more accurate the approximations are.
    
    compute_grad : bool, optional
        Whether to approximate the gradient of the log-partition function; default is True.

    print_error : bool, optional
        Whether to print the error in the batch Monte Carlo method; default is False.

    Returns
    -------
    float
        The approximation of the partition function evaluated at coef.
    
    numpy.ndarray
        The approximation of the gradient of the log-partition function evaluated at coef;
        only returns when compute_grad is True.

    """
    
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)

    if len(coef.shape) == 1:
        coef = coef.reshape(-1, 1)
    
    if basis_function.basisfunction_name == 'Polynomial':
        N = basis_function.degree
    else:
        N = basis_function.landmarks.shape[0]
        
    ###########################################################################
    # estimate the normalizing constant
    # first drawing
    mc_samples = base_density.sample(batch_size)
    mc_kernel_matrix = basis_function.basisfunction_eval(mc_samples)
    unnorm_density_part = np.exp(np.matmul(mc_kernel_matrix.T, coef))
    avg_norm_const = np.mean(unnorm_density_part)
    sq_norm_const = np.sum(unnorm_density_part ** 2)
    
    error_norm = np.sqrt(sq_norm_const / batch_size - avg_norm_const ** 2) / np.sqrt(batch_size)
    
    if print_error:
        print('normalizing constant error = {error:.7f}'.format(error=error_norm))
    
    batch_cnt = 1
    
    while error_norm > tol_param:
        
        # another draw
        mc_samples = base_density.sample(batch_size)
        mc_kernel_matrix = basis_function.basisfunction_eval(mc_samples)
        unnorm_density_part = np.exp(np.matmul(mc_kernel_matrix.T, coef))
        avg_norm_const2 = np.mean(unnorm_density_part)
        sq_norm_const += np.sum(unnorm_density_part ** 2)
        
        # update Monte Carlo estimation
        avg_norm_const = (avg_norm_const * batch_cnt + avg_norm_const2) / (batch_cnt + 1)
        
        error_norm = (np.sqrt(sq_norm_const / (batch_size * (batch_cnt + 1)) - avg_norm_const ** 2) /
                      np.sqrt(batch_size * (batch_cnt + 1)))
    
        batch_cnt += 1
        
        if print_error:
            print('normalizing constant error = {error:.7f}'.format(error=error_norm))
    
    normalizing_const = avg_norm_const
    print(batch_cnt)
    
    if compute_grad:
        
        print("#" * 45 + "\nApproximating the gradient of the log-partition now.")
        
        mc_samples = base_density.sample(batch_size)
        mc_kernel_matrix = basis_function.basisfunction_eval(mc_samples)
        density_part = np.exp(np.matmul(mc_kernel_matrix.T, coef).flatten()) / normalizing_const
        grad_est = (np.array([np.mean(mc_kernel_matrix[l1, :] * density_part)
                              for l1 in range(N)]).astype(np.float64).reshape(1, -1)[0])
        sq_grad_est = (np.array([(mc_kernel_matrix[l1, :] * density_part) ** 2
                                 for l1 in range(N)]).astype(np.float64).reshape(1, -1)[0])
        
        error_grad = np.sqrt(np.sum(np.mean(sq_grad_est, axis=0) - grad_est ** 2)) / np.sqrt(batch_size)
        
        if print_error:
            print('gradient error = {error:.7f}'.format(error=error_grad))
        
        batch_cnt = 1
        
        while error_grad > tol_param:
            
            # another draw
            mc_samples = base_density.sample(batch_size)
            mc_kernel_matrix = basis_function.basisfunction_eval(mc_samples)
            density_part = np.exp(np.matmul(mc_kernel_matrix.T, coef).flatten()) / normalizing_const
            grad_est2 = np.array([np.mean(mc_kernel_matrix[l1, :] * density_part)
                                  for l1 in range(N)]).astype(np.float64).reshape(1, -1)[0]
            sq_grad_est += (np.array([(mc_kernel_matrix[l1, :] * density_part) ** 2
                                      for l1 in range(N)]).astype(np.float64).reshape(1, -1)[0])
            
            grad_est = (grad_est * batch_cnt + grad_est2) / (batch_cnt + 1)
            
            error_grad = (np.sqrt(np.sum(np.mean(sq_grad_est, axis=0) / (batch_cnt + 1) - grad_est ** 2)) /
                          np.sqrt(batch_size * (batch_cnt + 1)))
            
            batch_cnt += 1
            
            if print_error:
                print('gradient error = {error:.7f}'.format(error=error_grad))
    
    if not compute_grad:
        return normalizing_const
    else:
        return normalizing_const, grad_est


def batch_montecarlo_params(mc_batch_size=1000, mc_tol=1e-2):
    
    """
    Returns a dictionary of parameters for the batch Monte Carlo method
    in approximating the log-partition function and its gradient.

    Parameters
    ----------
    mc_batch_size : int
        The batch size in the batch Monte Carlo method; default is 1000.

    mc_tol : float
        The floating point number below which sampling in the batch Monte Carlo is terminated; default is 1e-2.

    Returns
    -------
    dict
        The dictionary containing both supplied parameters.

    """

    mc_batch_size = int(mc_batch_size)
    
    output = {"mc_batch_size": mc_batch_size,
              "mc_tol": mc_tol}

    return output


def negloglik_optalgo_params(start_pt, step_size=0.01, max_iter=1e2, rel_tol=1e-5, abs_tol=1e-5):
    
    """
    Returns a dictionary of parameters used in minimizing the negative log-likelihood loss function
    by using the gradient descent algorithm.

    Parameters
    ----------
    start_pt : numpy.ndarray
        The starting point of the gradient descent algorithm to minimize
        the negative log-likelihood loss function.

    step_size : float or list or numpy.ndarray
        The step size used in the gradient descent algorithm; default is 0.01.

    max_iter : int
        The maximal number of iterations in the gradient descent algorithm; default is 100.

    rel_tol : float
        The relative tolerance parameter to terminate the gradient descent algorithm in minimizing
        the negative log-likelihood loss function; default is 1e-5.
    
    abs_tol : float
        The absolute tolerance parameter to terminate the gradient descent algorithm in minimizing
        the negative log-likelihood loss function; default is 1e-5.

    Returns
    -------
    dict
        The dictionary containing all supplied parameters.

    """

    assert step_size > 0., 'step_size must be strictly positive.'
    assert rel_tol > 0., 'rel_tol must be strictly positive.'
    assert abs_tol > 0., 'abs_tol must be strictly positive.'

    max_iter = int(max_iter)
    
    output = {"start_pt": start_pt,
              "step_size": step_size,
              "max_iter": max_iter,
              "rel_tol": rel_tol,
              "abs_tol": abs_tol}

    return output


def negloglik_finexpfam_coef(data, basis_function, base_density, optalgo_params, batchmc_params,
                             batch_mc=True, print_error=True):

    """
    Returns the natural parameter that minimizes the negative log-likelihood loss function
    in a finite-dimensional exponential family.
    The underlying minimization algorithm is the gradient descent algorithm.

    Parameters
    ----------
    data : numpy.ndarray
        The array of observations whose probability density function is to be estimated.

    basis_function : basis_function object
        The basis function used to estimate the probability density function.
        __type__ must be 'basis_function'.

    base_density : base_density object
        The base density function used to estimate the probability density function.
        __type__ must be 'base_density'.

    optalgo_params : dict
        The dictionary of parameters to control the gradient descent algorithm.
        Must be returned from the function negloglik_optalgo_params.

    batchmc_params : dict
        The dictionary of parameters to control the batch Monte Carlo method
        to approximate the partition function and the gradient of the log-partition function.
        Must be returned from the function batch_montecarlo_params.

    batch_mc : bool, optional
        Whether to use the batch Monte Carlo method with the termination criterion
        being the relative difference of two consecutive approximations; default is True.
        If it is False, the batch Monte Carlo method with the termination criterion
        being the standard deviation of the approximations will be used.
        
    print_error : bool, optional
        Whether to print the error of the gradient descent algorithm at each iteration; default is True.

    Returns
    -------
    numpy.ndarray
        An array of the natural parameter in the negative log-likelihood density estimate.

    """

    if len(data.shape) == 1:
        data = data.reshape(-1, 1)

    N, d = data.shape

    # parameters associated with gradient descent algorithm
    start_pt = optalgo_params['start_pt']
    step_size = optalgo_params['step_size']
    max_iter = optalgo_params['max_iter']
    rel_tol = optalgo_params['rel_tol']
    abs_tol = optalgo_params['abs_tol']

    if not isinstance(step_size, float):
        raise TypeError(("The type of step_size in optalgo_params should be float, but got {}".format(
            type(step_size))))

    if step_size <= 0.:
        raise ValueError("The step_size in optalgo_params must be strictly positive, but got {}.".format(
            step_size))

    # parameters associated with batch Monte Carlo estimation
    mc_batch_size = batchmc_params['mc_batch_size']
    mc_tol = batchmc_params['mc_tol']

    # the gradient of the loss function is
    # nabla L (alpha) = nabla A (alpha) - (1 / n) gram_matrix boldone_n
    # the gradient descent update is
    # new_iter = current_iter - step_size * nabla L (alpha)

    current_iter = start_pt.reshape(-1, 1)

    # compute the gradient of the log-partition function at current_iter
    if batch_mc:

        mc_output1, mc_output2 = negloglik_finexpfam_grad_logpar_batchmc(
            data=data,
            basis_function=basis_function,
            base_density=base_density,
            coef=current_iter,
            batch_size=mc_batch_size,
            tol_param=mc_tol,
            compute_grad=True,
            print_error=False)

        grad_logpar = mc_output2.reshape(-1, 1)

    else:

        mc_output1, mc_output2 = negloglik_finexpfam_grad_logpar_batchmc_se(
            data=data,
            basis_function=basis_function,
            base_density=base_density,
            coef=current_iter,
            batch_size=mc_batch_size,
            tol_param=mc_tol,
            compute_grad=True,
            print_error=False)

        grad_logpar = mc_output2.reshape(-1, 1)
        
    # form the Gram matrix
    gram = basis_function.basisfunction_eval(data)
    grad_term2 = gram.mean(axis=1, keepdims=True)

    # compute the gradient of the loss function at current_iter
    current_grad = grad_logpar - grad_term2

    # compute the updated iter
    new_iter = current_iter - step_size * current_grad

    # compute the error of the first update
    grad0_norm = np.linalg.norm(current_grad, 2)
    grad_new_norm = grad0_norm
    error = grad0_norm / grad0_norm
    # np.linalg.norm(new_iter - current_iter, 2) / (np.linalg.norm(current_iter, 2) + 1e-1)

    iter_num = 1

    if print_error:
        print("Iter = {iter_num}, GradNorm = {gradnorm}, Relative Error = {error}".format(
            iter_num=iter_num, gradnorm=grad0_norm, error=error))

    while error > rel_tol and grad_new_norm > abs_tol and iter_num < max_iter:

        current_iter = new_iter

        # compute the gradient at current_iter
        if batch_mc:

            mc_output1, mc_output2 = negloglik_finexpfam_grad_logpar_batchmc(
                data=data,
                basis_function=basis_function,
                base_density=base_density,
                coef=current_iter,
                batch_size=mc_batch_size,
                tol_param=mc_tol,
                compute_grad=True,
                print_error=False)

            grad_logpar = mc_output2.reshape(-1, 1)

        else:

            mc_output1, mc_output2 = negloglik_finexpfam_grad_logpar_batchmc_se(
                data=data,
                basis_function=basis_function,
                base_density=base_density,
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


def evaluate_negloglik_loss(data, new_data, basis_function, base_density, coef, batchmc_params, batch_mc=True):
    
    """
    Evaluates the negative log-likelihood loss function in a finite-dimensional exponential family.
    
    Parameters
    ----------
    data : numpy.ndarray
        The array of observations whose probability density function is to be estimated.
    
    new_data : numpy.ndarray
        The array of observations using which the negative log-likelihood loss function is evaluated.
    
    basis_function : basis_function object
        The basis function used to estimate the probability density function.
        __type__ must be 'basis_function'.
        
    base_density : base_density object
        The base density function used to estimate the probability density function.
        __type__ must be 'base_density'.
        
    coef : numpy.ndarray
        The natural parameter at which the score matching loss function is to be evaluated.
        
    batchmc_params : dict
        The dictionary of parameters to control the batch Monte Carlo method
        to approximate the partition function and the gradient of the log-partition function.
        Must be returned from the function batch_montecarlo_params.
        
    batch_mc : bool, optional
        Whether to use the batch Monte Carlo method with the termination criterion
        being the relative difference of two consecutive approximations; default is True.
        If it is False, the batch Monte Carlo method with the termination criterion
        being the standard deviation of the approximations will be used.
        
    Returns
    -------
    float
        The value of the negative log-likelihood loss function evaluated at coef.
    
    """
    
    new_data = check_data_type(new_data)
    new_data = check_data_dim(new_data)
    n, d1 = new_data.shape
    
    data = check_data_type(data)
    data = check_data_dim(data)
    N, d = data.shape
    
    assert d == d1, "The dimensionality of data and new_data are not the same."
    
    # parameters associated with batch Monte Carlo estimation
    mc_batch_size = batchmc_params["mc_batch_size"]
    mc_tol = batchmc_params["mc_tol"]
    
    # compute A(coef)
    if batch_mc:
        
        mc_output1 = negloglik_finexpfam_grad_logpar_batchmc(
            data=data,
            basis_function=basis_function,
            base_density=base_density,
            coef=coef,
            batch_size=mc_batch_size,
            tol_param=mc_tol,
            compute_grad=False,
            print_error=False)
    
    else:
        
        mc_output1 = negloglik_finexpfam_grad_logpar_batchmc_se(
            data=data,
            basis_function=basis_function,
            base_density=base_density,
            coef=coef,
            batch_size=mc_batch_size,
            tol_param=mc_tol,
            compute_grad=False,
            print_error=False)
        
    norm_const = mc_output1
    Af = np.log(norm_const)
    
    # compute (1 / n) \sum_{j=1}^N \innerp{coef}{T(Y_j)}, where Y_j is the j-th row of new_data
    basis_mat_new = basis_function.basisfunction_eval(new_data)
    avg_fx = np.mean(np.matmul(basis_mat_new.T, coef))
    
    loss_val = Af - avg_fx
    
    return loss_val


def negloglik_optparam(data, basis_function_name, base_density,
                       param_cand, k_folds, print_error, optalgo_params,
                       batchmc_params, save_dir=None, save_info=False, batch_mc=True):
    
    """
    Selects the optimal hyperparameter in the negative log-likelihood density estimation
    using k-fold cross validation and computes the coefficient vector at this optimal hyperparameter.

    Parameters
    ----------
    data : numpy.ndarray
        The array of observations whose probability density function is to be estimated.

    basis_function_name : str
        The type of the basis functions used to estimate the probability density function.
        Must be one of 'Gaussian', 'RationalQuadratic', 'Logistic', 'Triweight', and 'Sigmoid'.
        
    base_density : base_density object
        The base density function used to estimate the probability density function.
        __type__ must be 'base_density'.

    param_cand : list or 1-dimensional numpy.ndarray
        The list of hyperparameter candidates.

    k_folds : int
        The number of folds for cross validation.

    print_error : bool
        Whether to print the error of the gradient descent algorithm at each iteration.

    optalgo_params : dict
        The dictionary of parameters to control the gradient descent algorithm.
        Must be returned from the function negloglik_optalgo_params.

    batchmc_params : dict
        The dictionary of parameters to control the batch Monte Carlo method
        to approximate the partition function and the gradient of the log-partition function.
        Must be returned from the function batch_montecarlo_params.

    save_dir : str, optional
        The directory path to which the estimation information is saved;
        only works when save_info is True; default is None.

    save_info : bool, optional
        Whether to save the estimation information, including the values of negative log-likelihood
        loss function of each fold and the coefficient vector at the optimal hyperparameter, to a local file;
        default is False.

    batch_mc : bool, optional
        Whether to use the batch Monte Carlo method with the termination criterion
        being the relative difference of two consecutive approximations; default is True.
        If it is False, the batch Monte Carlo method with the termination criterion
        being the standard deviation of the approximations will be used.
        
    Returns
    -------
    dict
        A dictionary containing opt_param, the optimal hyperparameter, and
        opt_coef, the coefficient vector at the optimal hyperparameter.

    """
    
    check_basedensity(base_density)
    
    if basis_function_name not in ['Gaussian', 'RationalQuadratic', 'Logistic', 'Triweight', 'Sigmoid']:
        raise NotImplementedError(f"The basis function is {basis_function_name}, which has not been implemented.")
    
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    
    data = check_data_type(data)
    data = check_data_dim(data)
    N, d = data.shape
    
    # check the validity of param_cand
    # if basis_function_name == 'Polynomial':
    #
    #     param_cand_int = np.array(param_cand).flatten()
    #     if param_cand_int.dtype != int:
    #         raise ValueError("For the polynomial basis function, the param_cand should only contain integers.")
    #     if np.any(param_cand_int < 0):
    #         raise ValueError("There exists at least one element in param_cand whose value is negative. Please modify.")
    #
    # else:
    #
    param_cand = np.array(param_cand).flatten()
    if np.any(param_cand < 0.):
        raise ValueError("There exists at least one element in param_cand whose value is negative. Please modify.")

    n_param = len(param_cand)
    
    # check the step size
    step_size = optalgo_params['step_size']
    if isinstance(step_size, float):
        
        warn_msg = ("The step_size in optalgo_params is a float, and will be used in computing "
                    "density estimates for all {} different hyperparameter values in param_cand."
                    "It is better to supply a list or numpy.ndarray for step_size.").format(n_param)
        
        print(warn_msg)
        
        step_size = np.array([step_size] * n_param)
    
    elif isinstance(step_size, list):
        
        step_size = np.array(step_size)
    
    if len(step_size) != n_param:
        raise ValueError("The length of step_size in optalgo_params is not the same as that of param_cand.")
    
    folds_i = np.random.randint(low=0, high=k_folds, size=N)
    
    nll_scores = np.zeros((n_param,), dtype=np.float64)
    
    if save_info:
        f_log = open('%s/log.txt' % save_dir, 'w')
    
    for j in range(n_param):
        
        # initialize the loss score
        score = 0.
        current_param = param_cand[j]
        
        print("Parameter value " + str(j) + ": " + str(current_param))
        
        if save_info:
            f_log.write('parameter: %.8f, ' % current_param)
        
        for i in range(k_folds):
            # data split
            train_data = data[folds_i != i, ]
            test_data = data[folds_i == i, ]
            
            if basis_function_name == 'Gaussian':
                
                basis_function_sub = GaussianBasisFunction(
                    landmarks=train_data,
                    bw=current_param)
            
            elif basis_function_name == 'RationalQuadratic':
                
                basis_function_sub = RationalQuadraticBasisFunction(
                    landmarks=train_data,
                    bw=current_param)
            
            elif basis_function_name == 'Logistic':
                
                basis_function_sub = LogisticBasisFunction(
                    landmarks=train_data,
                    bw=current_param)
            
            elif basis_function_name == 'Triweight':
                
                basis_function_sub = TriweightBasisFunction(
                    landmarks=train_data,
                    bw=current_param)
            
            elif basis_function_name == 'Sigmoid':
                
                basis_function_sub = SigmoidBasisFunction(
                    landmarks=train_data,
                    bw=current_param)
            
            # compute the coefficient vector for the given hyperparameter
            train_algo_control = negloglik_optalgo_params(
                start_pt=np.zeros((train_data.shape[0], 1), dtype=np.float64),
                step_size=float(step_size[j]),
                max_iter=optalgo_params["max_iter"],
                rel_tol=optalgo_params["rel_tol"])
            
            coef = negloglik_finexpfam_coef(
                data=train_data,
                basis_function=basis_function_sub,
                base_density=base_density,
                optalgo_params=train_algo_control,
                batchmc_params=batchmc_params,
                batch_mc=batch_mc,
                print_error=print_error)
            
            score += evaluate_negloglik_loss(
                data=train_data,
                new_data=test_data,
                basis_function=basis_function_sub,
                base_density=base_density,
                coef=coef,
                batchmc_params=batchmc_params,
                batch_mc=batch_mc)
        
        nll_scores[j, ] = score / k_folds
        if save_info:
            f_log.write('score: %.8f\n' % nll_scores[j,])
    
    if save_info:
        f_log.close()
    
    cv_result = {np.round(x, 5): np.round(y, 10) for x, y in zip(param_cand, nll_scores)}
    print("The cross validation scores are:\n" + str(cv_result))
    
    # find the optimal hyperparameter
    opt_param = param_cand[np.argmin(nll_scores)]
    print("=" * 50)
    print("The optimal hyperparameter is {}.".format(opt_param))
    print("=" * 50 + "\nFinal run with the optimal hyperparameter.")
    
    # compute the coefficient vector at the optimal hyperparameter
    optalgo_params['step_size'] = float(step_size[np.argmin(nll_scores)])
    
    # form the basis function
    if basis_function_name == 'Gaussian':
        
        basis_function = GaussianBasisFunction(
            landmarks=data,
            bw=opt_param)
    
    elif basis_function_name == 'RationalQuadratic':
        
        basis_function = RationalQuadraticBasisFunction(
            landmarks=data,
            bw=opt_param)
    
    elif basis_function_name == 'Logistic':
        
        basis_function = LogisticBasisFunction(
            landmarks=data,
            bw=opt_param)
    
    elif basis_function_name == 'Triweight':
        
        basis_function = TriweightBasisFunction(
            landmarks=data,
            bw=opt_param)
    
    elif basis_function_name == 'Sigmoid':
        
        basis_function = SigmoidBasisFunction(
            landmarks=data,
            bw=opt_param)
    
    opt_coef = negloglik_finexpfam_coef(
        data=data,
        basis_function=basis_function,
        base_density=base_density,
        optalgo_params=optalgo_params,
        batchmc_params=batchmc_params,
        batch_mc=batch_mc,
        print_error=True)
    
    if save_info:
        f_optcoef = open('%s/negloglik_optcoef.npy' % save_dir, 'wb')
        np.save(f_optcoef, opt_coef)
        f_optcoef.close()
    
    output = {"opt_param": opt_param,
              "opt_coef": opt_coef}
    
    return output
