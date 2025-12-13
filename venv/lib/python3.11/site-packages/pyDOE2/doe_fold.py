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

__all__ = ['fold']

def fold(H, columns=None):
    """
    Fold a design to reduce confounding effects.
    
    Parameters
    ----------
    H : 2d-array
        The design matrix to be folded.
    columns : array
        Indices of of columns to fold (Default: None). If ``columns=None`` is
        used, then all columns will be folded.
    
    Returns
    -------
    Hf : 2d-array
        The folded design matrix.
    
    Examples
    --------
    ::
    
    """
    H = np.array(H)
    assert len(H.shape)==2, 'Input design matrix must be 2d.'
    
    if columns is None:
        columns = range(H.shape[1])
    
    Hf = H.copy()
    
    for col in columns:
        vals = np.unique(H[:, col])
        assert len(vals)==2, 'Input design matrix must be 2-level factors only.'
        
        for i in range(H.shape[0]):
            Hf[i, col] = vals[0] if H[i, col]==vals[1] else vals[1]
    
    Hf = np.vstack((H, Hf))
    
    return Hf
        
    
