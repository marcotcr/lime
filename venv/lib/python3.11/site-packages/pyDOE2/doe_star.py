"""
This code was originally published by the following individuals for use with
Scilab:
    Copyright (C) 2012 - 2013 - Michael Baudin
    Copyright (C) 2012 - Maria Christopoulou
    Copyright (C) 2010 - 2011 - INRIA - Michael Baudin
    Copyright (C) 2009 - Yann Collette
    Copyright (C) 2009 - CEA - Jean-Marc Martinez
    
    website: forge.scilab.org/index.php/p/scidoe/sourcetree/master/macros

Much thanks goes to these individuals. It has been converted to Python by 
Abraham Lee.
"""

import numpy as np

def star(n, alpha='faced', center=(1, 1)):
    """
    Create the star points of various design matrices
    
    Parameters
    ----------
    n : int
        The number of variables in the design
    
    Optional
    --------
    alpha : str
        Available values are 'faced' (default), 'orthogonal', or 'rotatable'
    center : array
        A 1-by-2 array of integers indicating the number of center points
        assigned in each block of the response surface design. Default is 
        (1, 1).
    
    Returns
    -------
    H : 2d-array
        The star-point portion of the design matrix (i.e. at +/- alpha)
    a : scalar
        The alpha value to scale the star points with.
    
    Example
    -------
    ::
    
        >>> star(3)
        array([[-1.,  0.,  0.],
               [ 1.,  0.,  0.],
               [ 0., -1.,  0.],
               [ 0.,  1.,  0.],
               [ 0.,  0., -1.],
               [ 0.,  0.,  1.]])
               
    """
    # Star points at the center of each face of the factorial
    if alpha=='faced':
        a = 1
    elif alpha=='orthogonal':
        nc = 2**n  # factorial points
        nco = center[0]  # center points to factorial
        na = 2*n  # axial points
        nao = center[1]  # center points to axial design
        # value of alpha in orthogonal design
        a = (n*(1 + nao/float(na))/(1 + nco/float(nc)))**0.5
    elif alpha=='rotatable':
        nc = 2**n  # number of factorial points
        a = nc**(0.25)  # value of alpha in rotatable design
    else:
        raise ValueError('Invalid value for "alpha": {:}'.format(alpha))
    
    # Create the actual matrix now.
    H = np.zeros((2*n, n))
    for i in range(n):
        H[2*i:2*i+2, i] = [-1, 1]
    
    H *= a
    
    return H, a
