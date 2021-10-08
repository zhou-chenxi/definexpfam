from basis_function import *


class UnnormalizedDensityFinExpFam:
    
    """
    A class for computing un-normalized density functions.

    ...

    Attributes
    ----------
    data : numpy.ndarray
        The array of observations with which the density function is estimated.
    
    basis_function : basis_function object
        The basis function used to estimate the probability density function.
        __type__ must be 'basis_function'.
        
    base_density : base_density object
        The base density function used to estimate the probability density function.
        __type__ must be 'base_density'.
    
    coef : numpy.ndarray
        The array of coefficients for basis functions in the natural parameter in the estimated density function.
        
    dim : int
        The dimensionality of the data, i.e., the number of columns of the data.
    
    Methods
    -------
    density_eval(new_data)
        Evaluates the un-normalized density estimate in a finite-dimensional exponential family
        at new data.
        
    density_eval_1d(x)
        Evaluates the un-normalized density estimate in a finite-dimensional exponential family
        at a 1-dimensional data point x.
        
    """
    
    def __init__(self, data, basis_function, base_density, coef):
        
        """
        Parameters
        ----------
        data : numpy.ndarray
            The array of observations whose density function is to be estimated.
    
        basis_function : basis_function object
            The basis function used to estimate the probability density function.
            Must be instantiated from the classes with __type__ being 'basis_function'.
            
        base_density : base_density object
            The base density function used to estimate the probability density function.
            Must be instantiated from the classes with __type__ being 'base_density'.
    
        coef : numpy.ndarray
            The array of coefficients for the natural parameter in the density estimate.
            
        """

        self.data = data
        self.basis_function = basis_function
        self.base_density = base_density
        self.coef = coef.reshape(-1, 1)
        
        if len(data.shape) == 1:
            self.data = self.data.reshape(-1, 1)
            
        self.dim = self.data.shape[1]
    
    def density_eval(self, new_data):
        
        """
        Evaluates the un-normalized density estimate in a finite-dimensional exponential family
        at new data.
        
        Parameters
        ----------
        new_data : numpy.ndarray
            The array of data at which the un-normalized density estimate is to be evaluated.
        
        Returns
        -------
        numpy.ndarray
            The 1-dimensional array of the un-normalized density estimates at new_data.
        
        """
        
        if len(new_data.shape) == 1:
            new_data = new_data.reshape(-1, 1)
            
        n, d = new_data.shape
        
        if d != self.dim: 
            raise ValueError("The dimensionality of new_data does not match that of data.")
        
        baseden_part = self.base_density.baseden_eval(new_data).flatten()
        exp_part = np.exp(np.matmul(self.basis_function.basisfunction_eval(new_data).T, self.coef)).flatten()
        
        output = baseden_part * exp_part
        
        return output

    def density_eval_1d(self, x):
        
        """
        Evaluates the un-normalized density estimate in a finite-dimensional exponential family
        at a 1-dimensional data point x.

        Parameters
        ----------
        x : float
            A floating point number at which the un-normalized density estimate is to be evaluated.

        Returns
        -------
        float
            A floating point number of the un-normalized density estimate at x.
            
        """
        
        if self.basis_function.basisfunction_name == 'Polynomial':
            
            den = (self.base_density.baseden_eval_1d(x) *
                   np.exp(np.sum([self.coef[i] * x ** (i + 1) for i in range(self.basis_function.degree)])))
            
        elif self.basis_function.basisfunction_name in ['Gaussian', 'RationalQuadratic', 'Logistic', 'Triweight', 'Sigmoid']:
            
            n_obs = self.basis_function.landmarks.shape[0]
            
            den = (self.base_density.baseden_eval_1d(x) *
                   np.exp(np.sum([self.coef[i] * self.basis_function.basis_x_1d(self.data[i].item())(x)
                                  for i in range(n_obs)])))
        
        else:
            
            msg = (f'The basis function type you specified is '
                   f'{self.basis_function.basisfunction_name}, which has not been implemented yet.')
            
            raise NotImplementedError(msg)
    
        return den
