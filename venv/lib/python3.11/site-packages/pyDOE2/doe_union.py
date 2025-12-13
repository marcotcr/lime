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

def union(H1, H2):
    """
    Join two matrices by stacking them on top of each other.
    
    Parameters
    ----------
    H1 : 2d-array
        The matrix that goes on top of the new matrix
    H2 : 2d-array
        The matrix that goes on bottom of the new matrix
    
    Returns
    -------
    mat : 2d-array
        The new matrix that contains the rows of ``H1`` on top of the rows of
        ``H2``.
    
    Example
    -------
    ::
    
        >>> union(np.eye(2), -np.eye(2))
        array([[ 1.,  0.],
               [ 0.,  1.],
               [-1.,  0.],
               [ 0., -1.]])
               
    """
    H = np.r_[H1, H2]
    return H
