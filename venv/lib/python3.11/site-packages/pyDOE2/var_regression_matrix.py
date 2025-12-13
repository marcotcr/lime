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

def var_regression_matrix(H, x, model, sigma=1):
    """
    Compute the variance of the 'regression error'.
    
    Parameters
    ----------
    H : 2d-array
        The regression matrix
    x : 2d-array
        The coordinates to calculate the regression error variance at.
    model : str
        A string of tokens that define the regression model (e.g. 
        '1 x1 x2 x1*x2')
    sigma : scalar
        An estimate of the variance (default: 1).
    
    Returns
    -------
    var : scalar
        The variance of the regression error, evaluated at ``x``.
        
    """
    x = np.atleast_2d(x)
    H = np.atleast_2d(H)
    
    if x.shape[0]==1:
        x = x.T
    
    if np.rank(H)<(np.dot(H.T, H)).shape[0]:
        raise ValueError("model and DOE don't suit together")
    
    x_mod = build_regression_matrix(x, model)
    var = sigma**2*np.dot(np.dot(x_mod.T, np.linalg.inv(np.dot(H.T, H))), x_mod)
    return var
