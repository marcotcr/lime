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

def grep(haystack, needle):
    start = 0
    while True:
        start = haystack.find(needle, start)
        if start==-1:
            return
        yield start
        start += len(needle)

def build_regression_matrix(H, model, build=None):
    """
    Build a regression matrix using a DOE matrix and a list of monomials.
    
    Parameters
    ----------
    H : 2d-array
    model : str
    build : bool-array
    
    Returns
    -------
    R : 2d-array
    
    """
    ListOfTokens = model.split(' ')
    if H.shape[1]==1:
        size_index = len(str(H.shape[0]))
    else:
        size_index = len(str(H.shape[1]))
    
    if build is None:
        build = [True]*len(ListOfTokens)
    
    # Test if the vector has the wrong direction (lines instead of columns)
    if H.shape[0]==1:
        H = H.T
    
    # Collect the list of monomials
    Monom_Index = []
    for i in range(len(ListOfTokens)):
        if build[i]:
            Monom_Index += [grep(ListOfTokens, 'x' + str(0)*(size_index - \
            len(str(i))) + str(i))]
    
    Monom_Index = -np.sort(-Monom_Index)
    Monom_Index = np.unique(Monom_Index)
    
    if H.shape[1]==1:
        nb_var = H.shape[0]  # vector "mode": the number of vars is equal to the number of lines of H
        VectorMode = True
        
        for i in range(nb_var):
            for j in range(ListOfTokens.shape[0]):
                ListOfTokens[j] = ListOfTokens[j].replace(
                    'x' + str(0)*(size_index - len(str(i))) + str(i),
                    'H(' + str(i) + ')')
    else:
        nb_var = H.shape[0]  # matrix "mode": the number of vars is equal to the number of columns of H
        VectorMode = False
        
        for i in range(nb_var):
            for j in range(ListOfTokens.shape[0]):
                ListOfTokens[j] = ListOfTokens[j].replace(
                    'x' + str(0)*(size_index - len(str(i))) + str(i),
                    'H[i,' + str(i) + ')')
    
    # Now build the regression matrix
    if VectorMode:
        R = np.zeros((len(ListOfTokens), 1))
        for j in range(len(ListOfTokens)):
            R[j, 0] = eval(ListOfTokens[j])
    else:
        R = np.zeros((H.shape[0], len(ListOfTokens)))
        for i in range(H.shape[0]):
            for j in range(len(ListOfTokens)):
                R[i, j] = eval(ListOfTokens[j])
    
    return R
    
    
