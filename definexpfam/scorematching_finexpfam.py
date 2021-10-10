from basis_function import *
from base_density import *


def scorematching_finexpfam_coef(data, basis_function, base_density):

    """
    Returns the solution to minimizing the score matching loss function
    in finite-dimensional exponential family.

    Parameters
    ----------
    data : numpy.ndarray
        The array of observations whose density function is to be estimated.

    basis_function : kernel_function object
        The basis function of the canonical statistic used to estimate the probability density function.
        __type__ must be 'basis_function'.

    base_density : base_density object
        The base density function used to estimate the probability density function.
        __type__ must be 'base_density'.

    Returns
    -------
    numpy.ndarray
        An array of coefficients for the natural parameter in the score matching density estimate.

    """

    N, d = data.shape
    if basis_function.basisfunction_name == 'Polynomial':
        n_basis = basis_function.degree
    else: 
        n_basis = basis_function.landmarks.shape[0]

    # compute DT(X_i) DT(X_i)^\top
    DT = basis_function.basisfunction_deriv1(data)
    dt_prod_term = sum([np.matmul(DT[:, i].reshape(n_basis, d),
                                  DT[:, i].reshape(n_basis, d).T) for i in range(N)])

    # compute the matrix G, which involves the second derivative
    matG = basis_function.basisfunction_deriv2(data)
    sum_matG = np.sum(sum([matG[:, i].reshape(n_basis, d) for i in range(N)]), axis=1, keepdims=True)

    # compute DT and grad of log mu
    # compute the gradient of log mu at data
    # each row corresponds to one data point
    grad_logmu = np.array([base_density.logbaseden_deriv1(data, j).flatten() for j in range(d)]).T
    dt_baseden_term = sum([np.matmul(DT[:, i].reshape(n_basis, d), grad_logmu[[i]].T) for i in range(N)])

    b_term = sum_matG + dt_baseden_term

    coef = -np.linalg.lstsq(dt_prod_term, b_term, rcond=None)[0]

    return coef


def evaluate_scorematching_loss(data, new_data, basis_function, base_density, coef):
    
    
    
    data = check_data_type(data)
    data = check_data_dim(data)
    N, d = data.shape
    new_data = check_data_type(new_data)
    new_data = check_data_dim(new_data)
    N1, d1 = new_data.shape
    
    assert d1 == d, "The dimensionality of data and new_data are not the same."
    
    if basis_function.basisfunction_name == 'Polynomial':
        n_basis = basis_function.degree
    else:
        n_basis = basis_function.landmarks.shape[0]
    
    # squared first derivative
    # compute DT(X_i) DT(X_i)^\top
    DT = basis_function.basisfunction_deriv1(new_data)
    dt_prod_term = sum([np.matmul(DT[:, i].reshape(n_basis, d),
                                  DT[:, i].reshape(n_basis, d).T) for i in range(N1)])
    
    # compute DT and grad of log mu
    # compute the gradient of log mu at data
    # each row corresponds to one data point
    grad_logmu = np.array([base_density.logbaseden_deriv1(new_data, j).flatten() for j in range(d)]).T
    dt_baseden_term = sum([np.matmul(DT[:, i].reshape(n_basis, d), grad_logmu[[i]].T) for i in range(N1)])
    
    output1 = 0.5 * np.matmul(coef.T, np.matmul(dt_prod_term, coef))[0][0] + np.sum(
        coef.flatten() * dt_baseden_term.flatten())
    # np.matmul(coef.T, dt_baseden_term)[0][0]
    
    # second derivative term
    # compute the matrix G, which involves the second derivative
    matG = basis_function.basisfunction_deriv2(new_data)
    sum_matG = np.sum(sum([matG[:, i].reshape(n_basis, d) for i in range(N1)]), axis=1, keepdims=True)
    
    output2 = np.sum(coef.flatten() * sum_matG.flatten())
    # np.matmul(coef.T, sum_matG)[0][0]
    
    # return output1 / N1, output2 / N1, (output1 + output2) / N1
    return (output1 + output2) / N1


def scorematching_optparam(data, basis_function_name, base_density, param_cand, k_folds,
                           save_dir=None, save_info=False):
    """
    Selects the optimal hyperparameter (e.g., the bandwidth parameter in the Gaussian basis function,
    and the degree in the polynomial basis function) in the score matching density estimation
    using k-fold cross validation and computes the coefficient vector of basis functions
    at this optimal hyperparameter.

    Parameters
    ----------
    data : numpy.ndarray
        The array of observations whose density function is to be estimated.






    k_folds : int
        The number of folds for cross validation.
    save_dir : str
        The directory path to which the estimation information is saved; only works when save_info is True.
    save_info : bool, optional
        Whether to save the estimation information, including the values of score matching loss function of
        each fold and the coefficient vector at the optimal penalty parameter, to a local file;
        default is False.

    Returns
    -------
    dict
        A dictionary containing opt_para, the value of the optimal hyperparameter, and
        opt_coef, the coefficient vector of basis functions in the penalized score matching density estimate
        at the optimal penalty parameter.

    """
    
    if basis_function_name not in ['Polynomial', 'Gaussian', 'RationalQuadratic', 'Logistic', 'Triweight', 'Sigmoid']:
        raise NotImplementedError(f"The basis function is {basis_function_name}, which has not been implemented.")
    
    check_basedensity(base_density)
    
    data = check_data_type(data)
    data = check_data_dim(data)
    N, d = data.shape
    
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    
    # check the validity of param_cand
    if basis_function_name == 'Polynomial':
        
        param_cand_int = np.array(param_cand).flatten()
        if param_cand_int.dtype != int:
            raise ValueError("For the polynomial basis function, the param_cand should only contain integers.")
        if np.any(param_cand_int < 0):
            raise ValueError("There exists at least one element in param_cand whose value is negative. Please modify.")
    
    else:
        
        param_cand = np.array(param_cand).flatten()
        if np.any(param_cand < 0.):
            raise ValueError("There exists at least one element in param_cand whose value is negative. Please modify.")
    
    n_param = len(param_cand)
    
    folds_i = np.random.randint(low=0, high=k_folds, size=N)
    
    sm_scores = np.zeros((n_param,), dtype=np.float64)
    
    if save_info:
        f_log = open('%s/log.txt' % save_dir, 'w')
    
    for j in range(n_param):
        
        # initialize the sm score
        score = 0
        current_param = param_cand[j]
        
        print("Parameter value " + str(j) + ": " + str(current_param))
        
        if save_info:
            f_log.write('parameter: %.8f, ' % current_param)
        
        for i in range(k_folds):
            # data split
            train_data = data[folds_i != i, ]
            test_data = data[folds_i == i, ]
            
            if basis_function_name == 'Polynomial':
                
                basis_function_sub = PolynomialBasisFunction(
                    landmarks=train_data,
                    degree=current_param)
            
            elif basis_function_name == 'Gaussian':
                
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
            coef_vec = scorematching_finexpfam_coef(
                data=train_data,
                basis_function=basis_function_sub,
                base_density=base_density)
            
            score += evaluate_scorematching_loss(
                data=train_data,
                new_data=test_data,
                basis_function=basis_function_sub,
                base_density=base_density,
                coef=coef_vec)
        
        sm_scores[j, ] = score / k_folds
        if save_info:
            f_log.write('score: %.8f\n' % sm_scores[j,])
    
    cv_result = {np.round(x, 5): np.round(y, 10) for x, y in zip(param_cand, sm_scores)}
    print("The cross validation scores are:\n" + str(cv_result))
    
    # find the optimal regularization parameter
    opt_param = param_cand[np.argmin(sm_scores)]
    
    print("=" * 50)
    print("The optimal hyperparameter is {}.".format(opt_param))
    print("=" * 50 + "\nFinal run with the optimal hyperparameter value.")
    
    # form the basis function
    if basis_function_name == 'Polynomial':
        
        basis_function = PolynomialBasisFunction(
            landmarks=data,
            degree=opt_param)
    
    elif basis_function_name == 'Gaussian':
        
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
    
    opt_coef = scorematching_finexpfam_coef(
        data=data,
        basis_function=basis_function,
        base_density=base_density)
    
    if save_info:
        f_log.close()
    
    if save_info:
        f_optcoef = open('%s/scorematching_optcoef.npy' % save_dir, 'wb')
        np.save(f_optcoef, opt_coef)
        f_optcoef.close()
    
    output = {"opt_para": opt_param,
              "opt_coef": opt_coef}
    
    return output
