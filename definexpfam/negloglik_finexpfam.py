from basis_function import *
from base_density import *


def negloglik_finexpfam_grad_logpar_batchmc(data, basis_function, base_density, coef, batch_size, tol_param,
                                            normalizing_const_only=False, print_error=False):
    
    """
    Approximates the log-partition function, A, at coef and its gradient at coef.
    The approximation method used is the batch Monte Carlo method. Terminate the sampling process
    until the relative difference of two consecutive approximations is less than tol_param.

    Let Y_1, ..., Y_M be random samples from the base density.
    The log-partition function evaluated at coef, A(coef), is approximated by
    log ((1 / M) sum_{i=1}^M exp ( sum_{j=1}^m coef[j] T_j (Y_i) )),
    and the gradient of the log-partition function evaluated at coef is approximated by
    (1 / M) sum_{i=1}^M T_l (Y_i) exp ( sum_{j=1}^m coef[j] T_j (Y_i) - A(coef)), for all l = 1, ..., n.
    
    Parameters
    ----------
    data : numpy.ndarray
        The array of observations whose density function is to be estimated.
    
    basis_function : basis_function object
        The basis function used to estimate the probability density function.
        __type__ must be 'basis_function'.
        
    base_density : base_density object
        The base density function used to estimate the probability density function.
        __type__ must be 'base_density'.
        
    coef : numpy.ndarray
        The array of coefficients at which the log-partition function and its gradient are approximated.

    batch_size : int
        The batch size in the batch Monte Carlo method.

    tol_param : float
        The floating point number below which sampling in the batch Monte Carlo is terminated.
        The smaller the tol_param is, the more accurate the approximations are.
    
    normalizing_const_only : bool, optional
        Whether to ONLY approximate the log-partition function but not its gradient; default is False.

    print_error : bool, optional
        Whether to print the error in the batch Monte Carlo method; default is False.

    Returns
    -------
    float
        The approximation of the log-partition function evaluated at coef.
    
    numpy.ndarray
        The approximation of the gradient of the log-partition function evaluated at coef;
        only returns when normalizing_const_only is False.
        
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
    
    if not normalizing_const_only:
        
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
    
    if normalizing_const_only: 
        return norm_est_new
    else: 
        return norm_est_new, grad_est_new


def negloglik_finexpfam_grad_logpar_batchmc_se(data, basis_function, base_density, coef, batch_size, tol_param,
                                               normalizing_const_only=False, print_error=False):
    
    """
    Approximates the log-partition function, A, at coef and its gradient at coef.
    The approximation method used is the batch Monte Carlo method. Terminate the sampling process
    until the standard deviation of the approximations is less than tol_param.
    
    Let Y_1, ..., Y_M be random samples from the base density.
    The log-partition function evaluated at coef, A(coef), is approximated by
    log ((1 / M) sum_{i=1}^M exp ( sum_{j=1}^m coef[j] T_j (Y_i) )),
    and the gradient of the log-partition function evaluated at coef is approximated by
    (1 / M) sum_{i=1}^M T_l (Y_i) exp ( sum_{j=1}^m coef[j] T_j (Y_i) - A(coef)), for all l = 1, ..., n.
    
    Parameters
    ----------
    data : numpy.ndarray
        The array of observations whose density function is to be estimated.
    
    basis_function : basis_function object
        The basis function used to estimate the probability density function.
        __type__ must be 'basis_function'.
        
    base_density : base_density object
        The base density function used to estimate the probability density function.
        __type__ must be 'base_density'.
        
    coef : numpy.ndarray
        The array of coefficients at which the log-partition function and its gradient are approximated.

    batch_size : int
        The batch size in the batch Monte Carlo method.

    tol_param : float
        The floating point number below which sampling in the batch Monte Carlo is terminated.
        The smaller the tol_param is, the more accurate the approximations are.
    
    normalizing_const_only : bool, optional
        Whether to ONLY approximate the log-partition function but not its gradient; default is False.

    print_error : bool, optional
        Whether to print the error in the batch Monte Carlo method; default is False.

    Returns
    -------
    float
        The approximation of the log-partition function evaluated at coef.
    
    numpy.ndarray
        The approximation of the gradient of the log-partition function evaluated at coef;
        only returns when normalizing_const_only is False.

    """
    
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)

        if len(coef.shape) == 1:
            coef = coef.reshape(-1, 1)
        
    N, d = data.shape
    
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
    
    if not normalizing_const_only:
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
                
        print(batch_cnt)
    
    if normalizing_const_only:
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


def negloglik_optalgoparams(start_pt, step_size=0.01, max_iter=1e2, rel_tol=1e-5):
    
    """
    Returns a dictionary of parameters used in minimizing the (penalized) negative log-likelihood loss function
    by using the gradient descent algorithm.

    Parameters
    ----------
    start_pt : numpy.ndarray
        The starting point of the gradient descent algorithm to minimize
        the penalized negative log-likelihood loss function.

    step_size : float or list or numpy.ndarray
        The step size used in the gradient descent algorithm; default is 0.01.

    max_iter : int
        The maximal number of iterations in the gradient descent algorithm; default is 100.

    rel_tol : float
        The relative tolerance parameter to terminate the gradient descent algorithm in minimizing
        the penalized negative log-likelihood loss function; default is 1e-5.

    Returns
    -------
    dict
        The dictionary containing all supplied parameters.

    """

    max_iter = int(max_iter)
    
    output = {"start_pt": start_pt,
              "step_size": step_size,
              "max_iter": max_iter,
              "rel_tol": rel_tol}

    return output

def negloglik_finexpfam_coef(data, basis_function, base_density, optalgo_params, batchmc_params,
                             batch_mc=True, batch_mc_se=False, print_error=True):

    """
    Returns the solution to minimizing the negative log-likelihood loss function
    in a finite-dimensional exponential family.
    The underlying minimization algorithm is the gradient descent algorithm.

    Parameters
    ----------
    data : numpy.ndarray
        The array of observations whose density function is to be estimated.

    basis_function : basis_function object
        The basis function used to estimate the probability density function.
        __type__ must be 'basis_function'.

    base_density : base_density object
        The base density function used to estimate the probability density function.
        __type__ must be 'base_density'.

    optalgo_params : dict
        The dictionary of parameters to control the gradient descent algorithm.
        Must be returned from the function negloglik_penalized_optalgoparams.

    batchmc_params : dict
        The dictionary of parameters to control the batch Monte Carlo method
        to approximate the log-partition function and its gradient.
        Must be returned from the function batch_montecarlo_params.

    batch_mc : bool, optional
        Whether to use the batch Monte Carlo method with the termination criterion
        being the relative difference of two consecutive approximations; default is True.

    batch_mc_se : bool, optional
        Whether to use the batch Monte Carlo method with the termination criterion
        being the standard deviation of approximations; default is False.

    print_error : bool, optional
        Whether to print the error of the gradient descent algorithm at each iteration; default is True.

    Returns
    -------
    numpy.ndarray
        An array of coefficients for the natural parameter in the negative log-likelihood density estimate.

    """

    if len(data.shape) == 1:
        data = data.reshape(-1, 1)

    N, d = data.shape

    # parameters associated with gradient descent algorithm
    start_pt = optalgo_params["start_pt"]
    step_size = optalgo_params["step_size"]
    max_iter = optalgo_params["max_iter"]
    rel_tol = optalgo_params["rel_tol"]

    if not type(step_size) == float:
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
    if batch_mc:

        mc_output1, mc_output2 = negloglik_finexpfam_grad_logpar_batchmc(
            data=data,
            basis_function=basis_function,
            base_density=base_density,
            coef=current_iter,
            batch_size=mc_batch_size,
            tol_param=mc_tol,
            normalizing_const_only=False,
            print_error=False)

        grad_logpar = mc_output2.reshape(-1, 1)

    elif batch_mc_se:

        mc_output1, mc_output2 = negloglik_finexpfam_grad_logpar_batchmc_se(
            data=data,
            basis_function=basis_function,
            base_density=base_density,
            coef=current_iter,
            batch_size=mc_batch_size,
            tol_param=mc_tol,
            normalizing_const_only=False,
            print_error=False)

        grad_logpar = mc_output2.reshape(-1, 1)

    else:
        
        raise NotImplementedError(("In order to approximate the gradient of the log-partition function, "
                                   "exactly one of 'batch_mc' and 'batch_mc_se' must be set True."))

    # form the Gram matrix
    gram = basis_function.basisfunction_eval(data)
    grad_term2 = gram.mean(axis=1, keepdims=True)

    # compute the gradient of the loss function at current_iter
    current_grad = grad_logpar - grad_term2

    # compute the updated iter
    new_iter = current_iter - step_size * current_grad

    # compute the error of the first update
    grad0_norm = np.linalg.norm(current_grad, 2)
    error = grad0_norm / grad0_norm
    # np.linalg.norm(new_iter - current_iter, 2) / (np.linalg.norm(current_iter, 2) + 1e-1)

    iter_num = 1

    if print_error:
        print("Iter = {iter_num}, GradNorm = {gradnorm}, Relative Error = {error}".format(
            iter_num=iter_num, gradnorm=grad0_norm, error=error))

    while error > rel_tol and iter_num < max_iter:

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
                normalizing_const_only=False,
                print_error=False)

            grad_logpar = mc_output2.reshape(-1, 1)

        elif batch_mc_se:

            mc_output1, mc_output2 = negloglik_finexpfam_grad_logpar_batchmc_se(
                data=data,
                basis_function=basis_function,
                base_density=base_density,
                coef=current_iter,
                batch_size=mc_batch_size,
                tol_param=mc_tol,
                normalizing_const_only=False,
                print_error=False)

            grad_logpar = mc_output2.reshape(-1, 1)

        else:

            raise NotImplementedError(("In order to approximate the gradient of the log-partition function, "
                                       "exactly one of 'batch_mc' and 'batch_mc_se' must be set True."))

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
