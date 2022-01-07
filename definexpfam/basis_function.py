import numpy as np
from definexpfam.check import *


class BasisFunction:
    
    """
    A parent class for the basis functions.

    Methods
    -------
    __type__
        Set __type__ to be the basis_function.

    """
    
    def __init__(self):
        pass
    
    @staticmethod
    def __type__():
        
        return 'basis_function'


class PolynomialBasisFunction(BasisFunction):
    
    """
    A class for the polynomial basis functions.

    ...

    Attributes
    ----------
    degree : int
        The degree of the polynomial basis functions; must be a strictly positive integer.

    landmarks : numpy.ndarray
        The data with which the basis functions are constructed.
    
    basisfunction_name : str
        The name of the basis function, that is, 'Polynomial'.
        
    Methods
    -------
    basisfunction_eval(new_data)
        Evaluates of the polynomial basis functions at new_data.
        
    basisfunction_deriv1(new_data)
        Evaluates the first derivative of the polynomial basis functions at new_data.
        
    basisfunction_deriv2(new_data)
        Evaluates the second derivative of the polynomial basis functions at new_data.
    
    """
    
    def __init__(self, landmarks, degree):
        
        super().__init__()
        
        if not isinstance(degree, int):
            degree = int(degree)
        
        self.degree = degree
        landmarks = check_data_type(landmarks)
        landmarks = check_data_dim(landmarks)
        self.landmarks = landmarks
        self.basisfunction_name = 'Polynomial'
    
    def basisfunction_eval(self, new_data):
        
        """
        Evaluates the polynomial basis functions at new_data.

        Parameters
        ----------
        new_data : numpy.ndarray
            A new data array at which the polynomial basis functions are to be evaluated.

        Returns
        -------
        numpy.ndarray
            An array of shape (self.degree, self.new_data.shape[0])
            whose (i, j)-entry is new_data[j] ** i.

        """
        
        new_data = check_data_type(new_data)
        new_data = check_data_dim(new_data)
        
        output = [new_data.flatten() ** (i + 1) for i in range(self.degree)]
        
        return np.array(output)
    
    def basisfunction_deriv1(self, new_data):
        
        """
        Evaluates the first derivative of the polynomial basis functions at new_data.

        Parameters
        ----------
        new_data : numpy.ndarray
            A new data array at which the first derivatives of the polynomial basis functions are to be evaluated.
            
        Returns
        -------
        numpy.ndarray
            An array of shape (self.degree, self.new_data.shape[0])
            whose (i, j)-entry is i * new_data[j] ** (i - 1).

        """
        
        new_data = check_data_type(new_data)
        new_data = check_data_dim(new_data)
        
        output = [(i + 1) * new_data.flatten() ** i for i in range(self.degree)]
        
        return np.array(output)
    
    def basisfunction_deriv2(self, new_data):
        
        """
        Evaluates the second derivative of the polynomial basis functions at new_data.

        Parameters
        ----------
        new_data : numpy.ndarray
            A new data array at which the second derivatives of the polynomial basis functions are to be evaluated.

        Returns
        -------
        numpy.ndarray
            An array of shape (self.degree, self.new_data.shape[0])
            whose (i, j)-entry is i * (i - 1) * new_data[j] ** (i - 2).
            
        """
        
        new_data = check_data_type(new_data)
        new_data = check_data_dim(new_data)
        
        if self.degree == 1:
            
            output = np.zeros((1, new_data.shape[0]))
        
        elif self.degree == 2:
            
            output = np.concatenate((np.zeros((1, new_data.shape[0])),
                                     2. * np.ones((1, new_data.shape[0]))))
        
        else:
            
            output1 = np.zeros((1, new_data.shape[0]))
            output2 = 2. * np.ones((1, new_data.shape[0]))
            output3 = [(i + 1) * i * new_data.flatten() ** (i - 1) for i in range(2, self.degree)]
            output = np.concatenate((output1, output2, output3))
        
        return np.array(output)


class GaussianBasisFunction(BasisFunction):
    
    """
    A class for the Gaussian basis functions.

    ...

    Attributes
    ----------
    bw : float
        The bandwidth parameter in the Gaussian basis functions; must be strictly positive.
        
    landmarks : numpy.ndarray
        The data at which the Gaussian basis functions are centered.
    
    N : int
        The number of the Gaussian basis functions.
    
    d : int
        The dimensionality of landmarks.
    
    basisfunction_name : str
        The name of the basis function, that is, 'Gaussian'.

    Methods
    -------
    basisfunction_eval(new_data)
        Evaluates the Gaussian basis functions at new_data.
    
    basisfunction_deriv1(new_data)
        Evaluates the first partial derivative of the Gaussian basis functions at new_data.
        
    basisfunction_deriv2(new_data)
        Evaluates the second partial derivative of the Gaussian basis functions at new_data.
        
    basis_x_1d(loc)
        Returns a function that computes k (loc, y) at y, where k (x, y) = exp(- (x - y) ^ 2 / (2 * bw ^ 2)),
        both loc and y are 1-dimensional data points.

    """
    
    def __init__(self, landmarks, bw):
        super().__init__()
        
        if bw <= 0.:
            raise ValueError("The bw parameter must be strictly positive.")
        self.bw = bw
        
        landmarks = check_data_type(landmarks)
        landmarks = check_data_dim(landmarks)
        self.landmarks = landmarks
        self.N, self.d = self.landmarks.shape
        self.basisfunction_name = 'Gaussian'
    
    def basisfunction_eval(self, new_data):
        
        """
        Evaluates the Gaussian basis functions at new_data.

        Parameters
        ----------
        new_data : numpy.ndarray
            A new data array at which the Gaussian basis functions are to be evaluated.

        Returns
        -------
        numpy.ndarray
            An array of shape (self.landmarks.shape[0], self.new_data.shape[0])
            whose (i, j)-entry is exp(- (X_i - Y_j)^2 / (2 * bw ^ 2)),
            where X_i is the i-th row in self.landmarks, and Y_j is the j-th row in new_data.

        """
        
        new_data = check_data_type(new_data)
        new_data = check_data_dim(new_data)
        n, d1 = new_data.shape
        
        assert self.d == d1, "The dimensionality of new_data does not match that of data. "
        
        bw = self.bw
        
        tiled_data = np.tile(new_data, self.N).reshape(1, -1)
        tiled_land = np.tile(self.landmarks.reshape(1, -1), n)
        
        diff = - (tiled_data - tiled_land) ** 2 / (2 * bw ** 2)
        power = np.sum(np.vstack(np.split(diff, self.N * n, axis=1)), axis=1)
        power = power.reshape(n, self.N)
        
        gauss_part = np.exp(power)
        
        return gauss_part.T
    
    def basisfunction_deriv1(self, new_data):
        
        """
        Evaluates the first partial derivatives of the Gaussian basis functions at new_data.

        Parameters
        ----------
        new_data : numpy.ndarray
            A new data array at which the first derivatives of the Gaussian basis functions are to be evaluated.

        Returns
        -------
        numpy.ndarray
            An array of shape (self.landmarks.shape[0], self.new_data.shape[0])
            whose (i, j)-entry is exp(- (X_i - Y_j)^2 / (2 * bw ^ 2)) * (X_i - Y_j) / bw ^ 2,
            where X_i is the i-th row in self.landmarks, and Y_j is the j-th row in new_data.
        
        """
        
        new_data = check_data_type(new_data)
        new_data = check_data_dim(new_data)
        n, d1 = new_data.shape
        
        assert self.d == d1, "The dimensionality of new_data does not match that of data. "
        
        gauss_kernel = np.repeat(self.basisfunction_eval(new_data=new_data),
                                 repeats=self.d, axis=0)
        multi_gauss1 = np.repeat(self.landmarks.flatten(), n).reshape(self.N * self.d, n)
        multi_gauss2 = np.tile(new_data.T, (self.N, 1))
        output = gauss_kernel * (multi_gauss1 - multi_gauss2) / self.bw ** 2
        
        return output
    
    def basisfunction_deriv2(self, new_data):
        
        """
        Evaluates the second partial derivatives of the Gaussian basis functions at new_data.

        Parameters
        ----------
        new_data : numpy.ndarray
            A new data array at which the second derivatives of the Gaussian basis functions are to be evaluated.
            
        Returns
        -------
        numpy.ndarray
            An array of shape (self.landmarks.shape[0], self.new_data.shape[0])
            whose (i, j)-entry is exp(- (X_i - Y_j)^2 / (2 * bw ^ 2)) * ((X_i - Y_j) ^ 2 - bw ^ 2) / bw ^ 4,
            where X_i is the i-th row in self.landmarks, and Y_j is the j-th row in new_data.

        """
        
        new_data = check_data_type(new_data)
        new_data = check_data_dim(new_data)
        n, d1 = new_data.shape
        
        assert self.d == d1, "The dimensionality of new_data does not match that of data. "
        
        gauss_kernel = np.repeat(self.basisfunction_eval(new_data=new_data), repeats=self.d, axis=0)
        multi_gauss_part1 = np.repeat(self.landmarks.flatten(), n).reshape(self.N * self.d, n)
        multi_gauss_part2 = np.tile(new_data.T, (self.N, 1))
        multi_gauss_part = ((multi_gauss_part1 - multi_gauss_part2) ** 2 / self.bw ** 4 -
                            1 / self.bw ** 2)
        output = multi_gauss_part * gauss_kernel
        
        return output

    def basis_x_1d(self, loc):
    
        """
        Returns a function that computes k (loc, y) at y, where k (x, y) = exp(- (x - y) ^ 2 / (2 * bw ^ 2)),
        both loc and y are 1-dimensional data points.

        Parameters
        ----------
        loc : float or np.ndarray
            A floating point number or a data array of shape (1,).

        Returns
        -------
        function
            A function that computes k (loc, y) at y.

        """
        
        if isinstance(loc, np.ndarray):
            loc = loc.item()
    
        def output(x):
            y = np.exp(- (x - loc) ** 2 / (2 * self.bw ** 2))
            return y
    
        return output


class RationalQuadraticBasisFunction(BasisFunction):
    
    """
    A class for the rational quadratic basis function.

    ...

    Attributes
    ----------
    bw : float
        The bandwidth parameter in the rational quadratic basis functions; must be strictly positive.
    
    landmarks : numpy.ndarray
        The data at which the basis functions are centered.
    
    N : int
        The number of the rational quadratic basis functions.
    
    d : int
        The dimensionality of self.landmarks.

    basisfunction_name : str
        The name of the basis function, that is, 'RationalQuadratic'.

    Methods
    -------
    basisfunction_eval(new_data)
        Evaluates the rational quadratic basis functions at new_data.
    
    basisfunction_deriv1(new_data)
        Evaluates the first partial derivatives of the rational quadratic
        basis functions at new_data.

    basisfunction_deriv2(new_data)
        Evaluates the second partial derivatives of the rational quadratic
        basis functions at new_data.

    basis_x_1d(loc)
        Returns a function that computes k (loc, y) at y, where k (x, y) = 1 / (1 + (x - y) ^ 2 / bw ^ 2),
        both loc and y are 1-dimensional data points.
        
    """
    
    def __init__(self, landmarks, bw):
        super().__init__()
        
        if bw <= 0.:
            raise ValueError("The bw parameter must be strictly positive.")
        self.bw = bw
        
        landmarks = check_data_type(landmarks)
        landmarks = check_data_dim(landmarks)
        self.landmarks = landmarks
        self.N, self.d = self.landmarks.shape
        self.basisfunction_name = 'RationalQuadratic'
    
    def basisfunction_eval(self, new_data):
        
        """
        Evaluates the rational quadratic basis functions at new_data.
        
        Parameters
        ----------
        new_data : numpy.ndarray
            A new data array at which the rational quadratic basis functions are to be evaluated.

        Returns
        -------
        numpy.ndarray
            An array of shape (self.landmarks.shape[0], self.new_data.shape[1])
            whose (i, j)-entry is 1 / (1 + ((X_i - Y_j) ^ 2 / (bw ^ 2))),
            where X_i is the i-th row in self.landmarks, and Y_j is the j-th row in new_data.

        """
        
        new_data = check_data_type(new_data)
        new_data = check_data_dim(new_data)
        n, d1 = new_data.shape
        
        assert self.d == d1, "The dimensionality of new_data does not match that of data. "
        
        bw = self.bw
        
        tiled_data = np.tile(new_data, self.N).reshape(1, -1)
        tiled_land = np.tile(self.landmarks.reshape(1, -1), n)
        
        diff = (tiled_data - tiled_land) ** 2 / (bw ** 2)
        power = np.sum(np.vstack(np.split(diff, self.N * n, axis=1)), axis=1)
        power = power.reshape(n, self.N)
        
        rationalquad_part = (1. + power) ** (-1)
        
        return rationalquad_part.T
    
    def basisfunction_deriv1(self, new_data):
        
        """
        Evaluates the first partial derivatives of the rational quadratic
        basis functions at new_data.

        Parameters
        ----------
        new_data : numpy.ndarray
            A new data array at which the first derivatives of the rational quadratic basis functions
            are to be evaluated.

        Returns
        -------
        numpy.ndarray
            An array of shape (self.landmarks.shape[0], self.new_data.shape[0])
            whose (i, j)-entry is 2 * (1 + (X_i - Y_j) ^ 2 / (bw ^ 2)) ^ (-2) * (X_i - Y_j) / bw ^ 2,
            where X_i is the i-th row in self.landmarks, and Y_j is the j-th row in new_data.

        """
        
        new_data = check_data_type(new_data)
        new_data = check_data_dim(new_data)
        n, d1 = new_data.shape
        
        assert self.d == d1, "The dimensionality of new_data does not match that of data. "
        
        rq_kernel = np.repeat(self.basisfunction_eval(new_data=new_data), repeats=self.d, axis=0)
        multi_rq1 = np.repeat(self.landmarks.flatten(), n).reshape(self.N * self.d, n)
        multi_rq2 = np.tile(new_data.T, (self.N, 1))
        output = 2 * rq_kernel ** 2 * (multi_rq1 - multi_rq2) / self.bw ** 2
        
        return output
    
    def basisfunction_deriv2(self, new_data):
        
        """
        Evaluates the second partial derivatives of the rational quadratic
        basis functions at new_data.
        
        Parameters
        ----------
        new_data : numpy.ndarray
            A new data array at which the second derivatives of the rational quadratic basis functions
            are to be evaluated.
            
        Returns
        -------
        numpy.ndarray
            An array of shape (self.landmarks.shape[0], self.new_data.shape[0]) whose (i, j)-entry is
            8 * (1 + (X_i - Y_j) ^ 2 / (bw ^ 2)) ^ (-3) * (X_i - Y_j) ^ 2 / bw ^ 4 -
            2 * (1 + (X_i - Y_j) ^ 2 / (bw ^ 2)) ^ (-2) / bw ^ 2,
            where X_i is the i-th row in self.landmarks, and Y_j is the j-th row in new_data.

        """
        
        new_data = check_data_type(new_data)
        new_data = check_data_dim(new_data)
        n, d1 = new_data.shape
        
        assert self.d == d1, "The dimensionality of new_data does not match that of data. "
        
        rq_kernel = np.repeat(self.basisfunction_eval(new_data=new_data), repeats=self.d, axis=0)
        multi_rq_part1 = np.repeat(self.landmarks.flatten(), n).reshape(self.N * self.d, n)
        multi_rq_part2 = np.tile(new_data.T, (self.N, 1))
        multi_rq_part = (multi_rq_part1 - multi_rq_part2) ** 2
        output = -2 * rq_kernel ** 2 / self.bw ** 2 + 8 * multi_rq_part * rq_kernel ** 3 / self.bw ** 4
        
        return output

    def basis_x_1d(self, loc):
    
        """
        Returns a function that computes k (loc, y) at y, where k (x, y) = 1 / (1 + (x - y) ^ 2 / bw ^ 2),
        both loc and y are 1-dimensional data points.

        Parameters
        ----------
        loc : float or np.ndarray
            A floating point number or a data array of shape (1,).

        Returns
        -------
        function
            A function that computes k (loc, y) at y.

        """
        
        if isinstance(loc, np.ndarray):
            loc = loc.item()
    
        def output(x):
            
            y = 1. / (1 + (x - loc) ** 2 / self.bw ** 2)
            
            return y
    
        return output


class LogisticBasisFunction(BasisFunction):
    
    """
    A class for the logistic basis function.

    ...

    Attributes
    ----------
    bw : float
        The bandwidth parameter in the logistic basis functions; must be strictly positive.
    
    landmarks : numpy.ndarray
        The data at which the basis functions are centered.
    
    N : int
        The number of the basis functions.
    
    d : int
        The dimensionality of self.landmarks.
        
    basisfunction_name : str
        The name of the basis function, that is, 'Logistic'.

    Methods
    -------
    basisfunction_eval(new_data)
        Evaluates the logistic basis functions at new_data.
    
    basisfunction_deriv1(new_data)
        Evaluates the first partial derivatives of the logistic basis functions at new_data.
    
    basisfunction_deriv2(new_data)
        Evaluates the second derivatives of the logistic basis functions at new_data.
    
    basis_x_1d(loc)
        Returns a function that computes k (loc, y) at y, where
        k (x, y) = 1 / (exp(-(x - y) / (2 * self.bw)) + exp((x - y) / (2 * self.bw)))^2,
        both loc and y are 1-dimensional data points.

    """
    
    def __init__(self, landmarks, bw):
        super().__init__()
        
        if bw <= 0.:
            raise ValueError("The bw parameter must be strictly positive.")
        self.bw = bw
        
        landmarks = check_data_type(landmarks)
        landmarks = check_data_dim(landmarks)
        self.landmarks = landmarks
        self.N, self.d = self.landmarks.shape
        
        self.basisfunction_name = 'Logistic'
    
    def basisfunction_eval(self, new_data):
        
        """
        Evaluates the logistic basis functions at new_data.

        Parameters
        ----------
        new_data : numpy.ndarray
            A new data array at which the logistic basis functions are to be evaluated.

        Returns
        -------
        numpy.ndarray
            An array of shape (self.landmarks.shape[0], self.new_data.shape[1])
            whose (i, j)-entry is (sech(-(X_i - Y_j) / (2 * self.bw))) ^ 2 / 4,
            also equal to 1 / (exp(-(X_i - Y_j) / (2 * self.bw)) + exp((X_i - Y_j) / (2 * self.bw)))^2,
            where X_i is the i-th row in self.landmarks, and Y_j is the j-th row in new_data.

        """
        
        new_data = check_data_type(new_data)
        new_data = check_data_dim(new_data)
        n, d1 = new_data.shape
        
        assert self.d == d1, "The dimensionality of new_data does not match that of data. "
        
        bw = self.bw
        
        tiled_data = np.tile(new_data, self.N).reshape(1, -1)
        tiled_land = np.tile(self.landmarks.reshape(1, -1), n)
        
        diff = (tiled_data - tiled_land) / (2. * bw)
        power = np.sum(np.vstack(np.split(diff, self.N * n, axis=1)), axis=1)
        power = power.reshape(n, self.N)
        
        output = 1. / np.cosh(power) ** 2 / 4.
        
        return output.T
    
    def basisfunction_deriv1(self, new_data):
        
        """
        Returns the evaluations of the first partial derivatives of the logistic basis functions
        with respect to the second argument at new_data.
        
        Parameters
        ----------
        new_data : numpy.ndarray
            A new data array at which the first derivatives of the logistic basis functions are to be evaluated.
            
        Returns
        -------
        numpy.ndarray
            An array of shape (self.landmarks.shape[0], self.new_data.shape[1])
            whose (i, j)-entry is
            (exp((X_i - Y_j) / (2 * self.bw)) - exp(-(X_i - Y_j) / (2 * self.bw))) /
            (exp(-(X_i - Y_j) / (2 * self.bw)) + exp((X_i - Y_j) / (2 * self.bw)))^3 / self.bw,
            where X_i is the i-th row in self.landmarks, and Y_j is the j-th row in new_data.

        """
        
        new_data = check_data_type(new_data)
        new_data = check_data_dim(new_data)
        n, d1 = new_data.shape
        
        assert self.d == d1, "The dimensionality of new_data does not match that of data. "
        
        bw = self.bw
        
        tiled_data = np.tile(new_data, self.N).reshape(1, -1)
        tiled_land = np.tile(self.landmarks.reshape(1, -1), n)
        
        diff = (tiled_data - tiled_land) / (2. * bw)
        power = np.sum(np.vstack(np.split(diff, self.N * n, axis=1)), axis=1)
        power = power.reshape(n, self.N)
        
        sech_part = 1. / np.cosh(power)
        tanh_part = np.sinh(power) / np.cosh(power)
        output = - 2. * (sech_part ** 2) * tanh_part / bw / 8.
        
        return output.T
    
    def basisfunction_deriv2(self, new_data):
        
        """
        Evaluates the second derivatives of the logistic basis functions at new_data.
        
        Parameters
        ----------
        new_data : numpy.ndarray
            A new data array at which the second derivatives of the logistic basis functions are to be evaluated.

        Returns
        -------
        numpy.ndarray
            An array of shape (self.landmarks.shape[0], self.new_data.shape[1])
            whose (i, j)-entry is the second partial derivative of the logistic basis functions
            evaluated at (X_i, Y_j), where X_i is the i-th row in self.landmarks,
            and Y_j is the j-th row in new_data.

        """
        
        new_data = check_data_type(new_data)
        new_data = check_data_dim(new_data)
        n, d1 = new_data.shape
        
        assert self.d == d1, "The dimensionality of new_data does not match that of data. "
        
        bw = self.bw
        
        tiled_data = np.tile(new_data, self.N).reshape(1, -1)
        tiled_land = np.tile(self.landmarks.reshape(1, -1), n)
        
        diff = (tiled_data - tiled_land) / (2. * bw)
        power = np.sum(np.vstack(np.split(diff, self.N * n, axis=1)), axis=1)
        power = power.reshape(n, self.N)
        
        sech_part = 1. / np.cosh(power)
        tanh_part = np.sinh(power) / np.cosh(power)
        output = (- 2. * sech_part ** 4 + 4. * sech_part ** 2 * tanh_part ** 2) / bw ** 2 / 16.
        
        return output.T

    def basis_x_1d(self, loc):
    
        """
        Returns a function that computes k (loc, y) at y, where
        k (x, y) = 1 / (exp(-(x - y) / (2 * self.bw)) + exp((x - y) / (2 * self.bw)))^2,
        both loc and y are 1-dimensional data points.

        Parameters
        ----------
        loc : float or np.ndarray
            A floating point number or a data array of shape (1,).

        Returns
        -------
        function
            A function that computes k (loc, y) at y.

        """
        
        if isinstance(loc, np.ndarray):
            loc = loc.item()
    
        def output(x):
            
            y = 1 / (np.exp(-(loc - x) / (2 * self.bw)) + np.exp((loc - x) / (2 * self.bw))) ** 2
        
            return y
    
        return output


class TriweightBasisFunction(BasisFunction):
    
    """
    A class for the triweight basis function.

    ...

    Attributes
    ----------
    bw : float
        The bandwidth parameter in the triweight basis functions; must be strictly positive.
    
    landmarks : numpy.ndarray
        The data at which the basis functions are centered.
        
    N : int
        The number of the basis functions.
    
    d : int
        The dimensionality of self.landmarks.
    
    basisfunction_name : str
        The name of the basis function, that is, 'Triweight'.
    
    Methods
    -------
    basisfunction_eval(new_data)
        Evaluates the triweight basis functions at new_data.
        
    basisfunction_deriv1(new_data)
        Evaluates the first derivatives of the triweight basis functions at new_data.
        
    basisfunction_deriv2(new_data)
        Evaluates the second derivatives of the triweight basis functions at new_data.
    
    basis_x_1d(loc)
        Returns a function that computes k (loc, y) at y, where
        k (x, y) = (1 - (x - y) ^ 2 / self.bw ^ 2) ^ 3 if |x - y| <= self.bw, and k (x, y) = 0, otherwise,
        both loc and y are 1-dimensional data points.

    """
    
    def __init__(self, landmarks, bw):
        super().__init__()
        
        if bw <= 0.:
            raise ValueError("The bw parameter must be strictly positive.")
        self.bw = bw
        
        landmarks = check_data_type(landmarks)
        landmarks = check_data_dim(landmarks)
        self.landmarks = landmarks
        self.N, self.d = self.landmarks.shape
        
        self.basisfunction_name = 'Triweight'
    
    def basisfunction_eval(self, new_data):
        
        """
        Evaluates the triweight basis functions at new_data.

        Parameters
        ----------
        new_data : numpy.ndarray
            A new data array at which the triweight basis functions are to be evaluated.

        Returns
        -------
        numpy.ndarray
            An array of shape (self.landmarks.shape[0], self.new_data.shape[1])
            whose (i, j)-entry is (1 - (X_i - Y_j)^2 / self.bw ^ 2) ^ 3
            if |X_i - Y_j| is less than or equal to self.bw and is 0 otherwise,
            where X_i is the i-th row in self.landmarks, and Y_j is the j-th row in new_data.
            
        """
        
        new_data = check_data_type(new_data)
        new_data = check_data_dim(new_data)
        n, d1 = new_data.shape
        
        assert self.d == d1, "The dimensionality of new_data does not match that of data. "
        
        bw = self.bw
        
        tiled_data = np.tile(new_data, self.N).reshape(1, -1)
        tiled_land = np.tile(self.landmarks.reshape(1, -1), n)
        
        diff = ((tiled_data - tiled_land) / bw) ** 2
        diff = (1. - diff) * (1. - diff >= 0.)
        # power = np.sum(np.vstack(np.split(diff, self.N * n, axis=1)), axis=1)
        power = diff.reshape(n, self.N)
        
        output = power ** 3
        
        return output.T
    
    def basisfunction_deriv1(self, new_data):
        
        """
        Evaluates the first derivatives of the triweight basis functions at new_data.

        Parameters
        ----------
        new_data : numpy.ndarray
            A new data array at which the first derivatives of the triweight basis functions are to be evaluated.
            
        Returns
        -------
        numpy.ndarray
            An array of shape (self.landmarks.shape[0], self.new_data.shape[1])
            whose (i, j)-entry is 6 * (1 - (X_i - Y_j) ^ 2 / self.bw ^ 2) ^ 2 * (X_i - Y_j) / self.bw ^ 2
            if |X_i - Y_j| is less than or equal to self.bw and is 0 otherwise,
            where X_i is the i-th row in self.landmarks, and Y_j is the j-th row in new_data.
            
        """
        
        new_data = check_data_type(new_data)
        new_data = check_data_dim(new_data)
        n, d1 = new_data.shape
        
        assert self.d == d1, "The dimensionality of new_data does not match that of data. "
        
        bw = self.bw
        
        tiled_data = np.tile(new_data, self.N).reshape(1, -1)
        tiled_land = np.tile(self.landmarks.reshape(1, -1), n)
        
        diff = (tiled_data - tiled_land) / bw
        diffsq = 1. - diff ** 2
        output = -6. * diff * diffsq ** 2 / bw * (diffsq >= 0.)
        output = output.reshape(n, self.N)
        
        return output.T
    
    def basisfunction_deriv2(self, new_data):
        
        """
        Evaluates the second derivatives of the triweight basis functions at new_data.

        Parameters
        ----------
        new_data : numpy.ndarray
            A new data array at which the second derivatives of the triweight basis functions are to be evaluated.
            
        Returns
        -------
        numpy.ndarray
            An array of shape (self.landmarks.shape[0], self.new_data.shape[1])
            whose (i, j)-entry is
            -6 * (1 - 5 * (X_i - Y_j) ^ 2 / self.bw ^ 2) ^ 2) * (1 - (X_i - Y_j) ^ 2 / self.bw ^ 2) / self.bw ^ 2
            if |X_i - Y_j| is less than or equal to self.bw and is 0 otherwise,
            where X_i is the i-th row in self.landmarks, and Y_j is the j-th row in new_data.
            
        """
        
        new_data = check_data_type(new_data)
        new_data = check_data_dim(new_data)
        n, d1 = new_data.shape
        
        assert self.d == d1, "The dimensionality of new_data does not match that of data. "
        
        bw = self.bw
        
        tiled_data = np.tile(new_data, self.N).reshape(1, -1)
        tiled_land = np.tile(self.landmarks.reshape(1, -1), n)
        
        diff = (tiled_data - tiled_land) / bw
        diffsq = diff ** 2
        output = (24. * diffsq * (1. - diffsq) - 6. * (1. - diffsq) ** 2) / bw ** 2 * (diffsq <= 1.)
        output = output.reshape(n, self.N)
        
        return output.T

    def basis_x_1d(self, loc):
    
        """
        Returns a function that computes k (loc, y) at y, where
        k (x, y) = (1 - (x - y) ^ 2 / self.bw ^ 2) ^ 3 if |x - y| <= self.bw, and k (x, y) = 0, otherwise,
        both loc and y are 1-dimensional data points.

        Parameters
        ----------
        loc : float or np.ndarray
            A floating point number or a data array of shape (1,).

        Returns
        -------
        function
            A function that computes k (loc, y) at y.

        """
        
        if isinstance(loc, np.ndarray):
            loc = loc.item()
    
        def output(x):
            
            if np.abs(x - loc) <= self.bw:
                y = (1 - (loc - x) ** 2 / self.bw ** 2) ** 3
            else:
                y = 0
        
            return y
    
        return output


class SigmoidBasisFunction(BasisFunction):
    
    """
    A class for the sigmoid basis function.

    ...

    Attributes
    ----------
    bw : float
        The bandwidth parameter in the sigmoid basis functions; must be strictly positive.
    
    landmarks : numpy.ndarray
        The data at which the sigmoid basis functions are centered.
    
    N : int
        The number of the basis functions.
        
    d : int
        The dimensionality of landmarks.
    
    basisfunction_name : str
        The name of the basis function, that is, 'Sigmoid'.
        
    Methods
    -------
    basisfunction_eval(new_data)
        Evaluates the sigmoid basis functions at new_data.
        
    basisfunction_deriv1(new_data)
        Evaluates the first derivatives of the sigmoid basis functions at new_data.
        
    basisfunction_deriv2(new_data)
        Evaluates the second derivatives of the sigmoid basis functions at new_data.
        
    basis_x_1d(loc)
        Returns a function that computes k (loc, y) at y, where
        k (x, y) = 1 / (1 + exp ( - (x - y) / self.bw)),
        and both loc and y are 1-dimensional data points.

    """
    
    def __init__(self, landmarks, bw):
        super().__init__()
        
        if bw <= 0.:
            raise ValueError("The bw parameter must be strictly positive.")
        self.bw = bw
        
        landmarks = check_data_type(landmarks)
        landmarks = check_data_dim(landmarks)
        self.landmarks = landmarks
        self.N, self.d = self.landmarks.shape
        
        self.basisfunction_name = 'Sigmoid'
    
    def basisfunction_eval(self, new_data):
        
        """
        Evaluates the sigmoid basis functions at new_data.

        Parameters
        ----------
        new_data : numpy.ndarray
            A new data array at which the sigmoid basis functions are to be evaluated.
            
        Returns
        -------
        numpy.ndarray
            An array of shape (self.landmarks.shape[0], self.new_data.shape[1])
            whose (i, j)-entry is 1 / (1 + exp(-(x - y) / self.bw)),
            where X_i is the i-th row in self.landmarks, and Y_j is the j-th row in new_data.
            
        """
        
        new_data = check_data_type(new_data)
        new_data = check_data_dim(new_data)
        n, d1 = new_data.shape
        
        assert self.d == d1, "The dimensionality of new_data does not match that of data. "
        
        bw = self.bw
        
        tiled_data = np.tile(new_data, self.N).reshape(1, -1)
        tiled_land = np.tile(self.landmarks.reshape(1, -1), n)
        diff = -(tiled_data - tiled_land) / bw
        # power = np.sum(np.vstack(np.split(diff, self.N * n, axis=1)), axis=1)
        power = diff.reshape(n, self.N)
        
        output = 1. / (1. + np.exp(-power))
        
        return output.T
    
    def basisfunction_deriv1(self, new_data):
        
        """
        Evaluates the first derivatives of the sigmoid basis functions at new_data.

        Parameters
        ----------
        new_data : numpy.ndarray
            A new data array at which the first derivatives of the sigmoid basis function are to be evaluated.
            
        Returns
        -------
        numpy.ndarray
            An array of shape (self.landmarks.shape[0], self.new_data.shape[1])
            whose (i, j)-entry is - exp(-(X_i - Y_j) / self.bw) / (1 + exp(-(X_i - Y_j) / self.bw)) ^ 2 / self.bw,
            where X_i is the i-th row in self.landmarks, and Y_j is the j-th row in new_data.
            
        """
        
        new_data = check_data_type(new_data)
        new_data = check_data_dim(new_data)
        n, d1 = new_data.shape
        
        assert self.d == d1, "The dimensionality of new_data does not match that of data. "
        
        bw = self.bw
        
        tiled_data = np.tile(new_data, self.N).reshape(1, -1)
        tiled_land = np.tile(self.landmarks.reshape(1, -1), n)
        diff = -(tiled_data - tiled_land) / bw
        # power = np.sum(np.vstack(np.split(diff, self.N * n, axis=1)), axis=1)
        power = diff.reshape(n, self.N)
        
        output = -1. / (1. + np.exp(-power)) ** 2 * np.exp(-power) / bw
        
        return output.T
    
    def basisfunction_deriv2(self, new_data):
        
        """
        Evaluates the second derivatives of the sigmoid basis functions at new_data.
        
        Parameters
        ----------
        new_data : numpy.ndarray
            A new data array at which the second derivatives of the sigmoid basis functions are to be evaluated.
            
        Returns
        -------
        numpy.ndarray
            An array of shape (self.landmarks.shape[0], self.new_data.shape[1])
            whose (i, j)-entry is
            2 * exp(-2 * (x - y) / self.bw) / (1 + exp(-(x - y) / self.bw)) ^ 3 / self.bw ^ 2 -
                exp(-(x - y) / self.bw) / (1 + exp(-(x - y) / self.bw)) ^ 2 / self.bw ^ 2,
            where X_i is the i-th row in self.landmarks, and Y_j is the j-th row in new_data.
            
        """
        
        new_data = check_data_type(new_data)
        new_data = check_data_dim(new_data)
        n, d1 = new_data.shape
        
        assert self.d == d1, "The dimensionality of new_data does not match that of data. "
        
        bw = self.bw
        
        tiled_data = np.tile(new_data, self.N).reshape(1, -1)
        tiled_land = np.tile(self.landmarks.reshape(1, -1), n)
        diff = -(tiled_data - tiled_land) / bw
        # power = np.sum(np.vstack(np.split(diff, self.N * n, axis=1)), axis=1)
        power = diff.reshape(n, self.N)
        
        output1 = 2. * np.exp(- 2. * power) / (1. + np.exp(-power)) ** 3 / bw ** 2
        output2 = np.exp(-power) / (1. + np.exp(-power)) ** 2 / bw ** 2
        output = output1 - output2
        
        return output.T

    def basis_x_1d(self, loc):
    
        """
        Returns a function that computes k (loc, y) at y, where
        k (x, y) = 1 / (1 + exp(-(x - y) / self.bw)),
        both loc and y are 1-dimensional data points.

        Parameters
        ----------
        loc : float or np.ndarray
            A floating point number or a data array of shape (1,).

        Returns
        -------
        function
            A function that computes k (loc, y) at y.

        """

        if isinstance(loc, np.ndarray):
            loc = loc.item()
    
        def output(x):
        
            y = 1. / (1. + np.exp(-(loc - x) / self.bw))
            
            return y
    
        return output
