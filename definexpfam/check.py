import numpy as np

def check_data_type(data):
    
    """

    """
    
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    return data


def check_data_dim(data):
    
    """

    """
    
    dshape = data.shape
    
    if len(dshape) == 1:
        
        data = data.reshape(-1, 1)
    
    else:
        
        if dshape[1] != 1:
            raise ValueError(f'data should be 1-dimensional, but got {dshape[1]}-dimensional.')
    
    return data


def check_basedensity(base_density):

    """
    Check the type of base_density.

    base_density : base_density object
        The base density function used to estimate the probability density function.
        __type__ must be 'base_density'.

    """

    input_type = base_density.__type__()

    if input_type != 'base_density':

        raise TypeError('The __type__ of base_density should be base_density, but got {}'.format(input_type))

    else:

        pass


def check_basisfunction(basis_function):

    """
    Check the type of basis_function.

    basis_function : basis_function object
        The basis function used to estimate the probability density function.
        __type__ must be 'basis_function'.

    """

    input_type = basis_function.__type__()

    if input_type != 'basis_function':

        raise TypeError('The __type__ of basis_function should be basis_function, but got {}'.format(input_type))

    else:

        pass


def check_basisfunction_name(basis_function):

    """
    Check the type of basis_function.

    basis_function : kernel_function object
        The basis function used to estimate the probability density function,
        which must be one of 'Polynomial', 'Gaussian', 'RationalQuadratic', 'Logistic', 'Triweight', and 'Sigmoid'.

    """

    bf_name = basis_function.basisfunction_name
    if bf_name not in ['Polynomial', 'Gaussian', 'RationalQuadratic', 'Logistic', 'Triweight', 'Sigmoid']:
    
        raise NotImplementedError(f"The basis function is {bf_name}, which has not been implemented.")
    
    else:

        pass
    
# def check_samedata(kernel_function, base_density):
#
# 	"""
# 	Check whether the data in kernel_function and the data in base_density are identical.
#
# 	kernel_function : kernel_function object
# 		The kernel function used to estimate the probability density function.
# 		__type__ must be 'kernel_function'.
#
# 	base_density : base_density object
# 		The base density function used to estimate the probability density function.
# 		__type__ must be 'base_density'.
#
# 	"""
#
# 	kf_data = kernel_function.data
# 	bd_data = base_density.data
#
# 	if not np.allclose(kf_data, bd_data):
#
# 		raise ValueError(('The data from kernel_function and base_density are different. '
# 						  'Please double check the data input.'))
#
# 	else:
#
# 		pass