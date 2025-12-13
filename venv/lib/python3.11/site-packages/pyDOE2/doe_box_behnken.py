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
from pyDOE2.doe_factorial import ff2n
from pyDOE2.doe_repeat_center import repeat_center

__all__ = ['bbdesign']

def bbdesign(n, center=None):
    """
    Create a Box-Behnken design
    
    Parameters
    ----------
    n : int
        The number of factors in the design
    
    Optional
    --------
    center : int
        The number of center points to include (default = 1).
    
    Returns
    -------
    mat : 2d-array
        The design matrix
    
    Example
    -------
    ::
    
        >>> bbdesign(3)
        array([[-1., -1.,  0.],
               [ 1., -1.,  0.],
               [-1.,  1.,  0.],
               [ 1.,  1.,  0.],
               [-1.,  0., -1.],
               [ 1.,  0., -1.],
               [-1.,  0.,  1.],
               [ 1.,  0.,  1.],
               [ 0., -1., -1.],
               [ 0.,  1., -1.],
               [ 0., -1.,  1.],
               [ 0.,  1.,  1.],
               [ 0.,  0.,  0.],
               [ 0.,  0.,  0.],
               [ 0.,  0.,  0.]])
        
    """
    assert n>=3, 'Number of variables must be at least 3'
    
    # First, compute a factorial DOE with 2 parameters
    H_fact = ff2n(2)
    # Now we populate the real DOE with this DOE
    
    # We made a factorial design on each pair of dimensions
    # - So, we created a factorial design with two factors
    # - Make two loops
    Index = 0
    nb_lines = int((0.5*n*(n-1))*H_fact.shape[0])
    H = repeat_center(n, nb_lines)
    
    for i in range(n - 1):
        for j in range(i + 1, n):
            Index = Index + 1
            H[max([0, (Index - 1)*H_fact.shape[0]]):Index*H_fact.shape[0], i] = H_fact[:, 0]
            H[max([0, (Index - 1)*H_fact.shape[0]]):Index*H_fact.shape[0], j] = H_fact[:, 1]

    if center is None:
        if n<=16:
            points= [0, 0, 0, 3, 3, 6, 6, 6, 8, 9, 10, 12, 12, 13, 14, 15, 16]
            center = points[n]
        else:
            center = n
        
    H = np.c_[H.T, repeat_center(n, center).T].T
    
    return H
