import numpy as np 
from basis_function import *
from base_density import *


def evaluate_scorematching_loss(data, basis_function, base_density, coef): 
    
    N, d = data.shape
    
    if basis_function.basisfunction_name == 'Polynomial':
        n_basis = basis_function.degree
    else: 
        n_basis = basis_function.landmarks.shape[0]
    
    # squared first derivative 
    # compute DT(X_i) DT(X_i)^\top
    DT = basis_function.basisfunction_deriv1(data)
    dt_prod_term = sum([np.matmul(DT[:, i].reshape(n_basis, d),
                                  DT[:, i].reshape(n_basis, d).T) for i in range(N)])

    # compute DT and grad of log mu
    # compute the gradient of log mu at data
    # each row corresponds to one data point
    grad_logmu = np.array([base_density.logbaseden_deriv1(data, j).flatten() for j in range(d)]).T
    dt_baseden_term = sum([np.matmul(DT[:, i].reshape(n_basis, d), grad_logmu[[i]].T) for i in range(N)])

    output1 = 0.5 * np.matmul(coef.T, np.matmul(dt_prod_term, coef))[0][0] + np.sum(coef.flatten() * dt_baseden_term.flatten())
    # np.matmul(coef.T, dt_baseden_term)[0][0]
    
    # second derivative term 
    # compute the matrix G, which involves the second derivative
    matG = basis_function.basisfunction_deriv2(data)
    sum_matG = np.sum(sum([matG[:, i].reshape(n_basis, d) for i in range(N)]), axis=1, keepdims=True)

    output2 = np.sum(coef.flatten() * sum_matG.flatten())
    # np.matmul(coef.T, sum_matG)[0][0]
    
    return output1 / N, output2 / N, (output1 + output2) / N
    