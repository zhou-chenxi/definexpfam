import sys
import matplotlib.pyplot as plt
from scipy import integrate
from definexpfam.unnormalized_density import *


def plot_density_1d_params(x_limit, y_limit, plot_pts_cnt=2000, figsize=(10, 10),
                           den_color='tab:blue', hist_color='tab:blue', bins='fd', hist_alpha=0.5, fontsize=20):
    
    """
    Specifies and returns the plotting parameters used in the function plot_density_1d.
    
    Parameters
    ----------
    x_limit : tuple
        The tuple to specify the plotting domain of the density estimate.
        Must be of length 2. Both components must be finite numbers.
            
    y_limit : tuple
        The tuple to specify the domain of the plot of density estimate in the vertical axis.
        Must be of length 2. Both components must be finite numbers.
        
    plot_pts_cnt : int, optional
        The number of points to be evaluated along the plot_domain to make a plot of density estimate;
        default is 2000.
        
    figsize : typle, optional
        The size of the plot of density estimate; default is (10, 10).
            
    den_color : str or tuple, optional
        The color for plotting the density estimate; default is 'tab:blue'.
        See details at https://matplotlib.org/3.1.0/tutorials/colors/colors.html.
        
    hist_color : str or tuple, optional
        The color for plotting the histogram; default is 'tab:blue'.
        See details at https://matplotlib.org/3.1.0/tutorials/colors/colors.html.
        
    bins : int or sequence or str, optional
        The bins used for plotting the histogram; default is 'fd', the Freedman–Diaconis rule.
        See details at https://matplotlib.org/3.3.3/api/_as_gen/matplotlib.pyplot.hist.html.
        
    hist_alpha : float, optional
        Set the alpha value used for blending in plotting the histogram; default is 0.5.
        
    fontsize : int, optional
        The font size in the plot; default is 20.
            
    Returns
    -------
    dict
        A dict containing all the plotting parameter inputs.
            
    """

    if len(x_limit) != 2:
        raise ValueError("The length of x_limit must be 2.")

    if len(y_limit) != 2:
        raise ValueError("The length of y_limit must be 2.")

    if np.inf in x_limit or -np.inf in x_limit:
        raise ValueError("x_limit contains non-finite values.")

    if np.inf in y_limit or -np.inf in y_limit:
        raise ValueError("y_limit contains non-finite values.")

    output = {'x_limit': x_limit,
              'y_limit': y_limit,
              'plot_pts_cnt': plot_pts_cnt,
              'figsize': figsize,
              'den_color': den_color,
              'hist_color': hist_color,
              'bins': bins,
              'hist_alpha': hist_alpha,
              'fontsize': fontsize}
    
    return output


def plot_density_1d(data, basis_function, base_density, coef, normalizing, method, x_label,
                    plot_kwargs, save_plot=False, save_dir=None, save_filename=None):

    """
    Plots the density estimate with the histogram over a bounded one-dimensional interval.
    
    Parameters
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
        The array of coefficients of basis functions in the estimated density function.
        
    normalizing : bool
        Whether to plot the normalized density estimate.
        
    method : str
        The density estimation method.
    
    x_label : str
        The label of the horizontal axis.
    
    plot_kwargs : dict
        The dict containing plotting parameters returned from the function plot_density_1d_params.
    
    save_plot : bool, optional
        Whether to save the plot of the density estimate as a local file; default is False.
    
    save_dir : str, optional
        The directory path to which the plot of the density estimate is saved;
        only works when save_plot is set to be True. Default is None.
    
    save_filename : str, optional
        The file name for the plot of the density estimate saved as a local file;
        only works when save_plot is set to be True. Default is None.
        
    Returns
    -------
    dict
        A dictionary of x_vals, the values of the horizontal axis for plotting, and
        den_vals, the values of the vertial axis for plotting.
    
    """
    
    if len(data.shape) != 1:
        data = data.reshape(-1, 1)
    
    if data.shape[1] != 1:
        raise ValueError("The data should be of 1 column.")
    
    coef = coef.reshape(-1, 1)
    
    plot_domain = [[plot_kwargs['x_limit'][0], plot_kwargs['x_limit'][1]]]
    plot_pts_cnt = plot_kwargs['plot_pts_cnt']
    
    unnorm = UnnormalizedDensityFinExpFam(
        data=data,
        basis_function=basis_function,
        base_density=base_density,
        coef=coef)
    
    unnorm_fun = unnorm.density_eval
    unnorm_fun_int = unnorm.density_eval_1d
    
    x0_cand = np.linspace(plot_domain[0][0], plot_domain[0][1], num=plot_pts_cnt).reshape(-1, 1)
    plot_val = unnorm_fun(x0_cand)

    if normalizing:
        norm_const, _ = integrate.nquad(unnorm_fun_int, base_density.domain, opts={'limit': 100})
        plot_val /= norm_const
        
    fig = plt.figure(figsize=plot_kwargs['figsize'])
    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax = fig.add_axes([left, bottom, width, height])

    plt.plot(x0_cand, plot_val, plot_kwargs['den_color'])
    plt.hist(data.flatten(),
             color=plot_kwargs['hist_color'],
             bins=plot_kwargs['bins'],
             range=plot_kwargs['x_limit'],
             density=True,
             alpha=plot_kwargs['hist_alpha'])
    # plt.plot(data, [0.01] * len(data), '|', color = 'k')
    
    ax.set_title('Density Plot (' + method + ')', fontsize=plot_kwargs['fontsize'])
    ax.set_xlabel(x_label, fontsize=plot_kwargs['fontsize'])
    if normalizing:
        ax.set_ylabel('density', fontsize=plot_kwargs['fontsize'])
    else:
        ax.set_ylabel('unnormalized density', fontsize=plot_kwargs['fontsize'])
    ax.set_xlim(plot_kwargs['x_limit'])
    ax.set_ylim(plot_kwargs['y_limit'])
    ax.tick_params(axis='both', labelsize=plot_kwargs['fontsize'])
    if save_plot: 
        plt.savefig(save_dir + save_filename + '.pdf')
    plt.show()
    
    return {"x_vals": x0_cand, "den_vals": plot_val}
