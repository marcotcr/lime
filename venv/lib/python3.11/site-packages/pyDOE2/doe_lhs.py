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
from scipy import spatial
from scipy import stats
from scipy import linalg
from numpy import ma

__all__ = ['lhs']


def lhs(n, samples=None, criterion=None, iterations=None, random_state=None, correlation_matrix = None):
    """
    Generate a latin-hypercube design

    Parameters
    ----------
    n : int
        The number of factors to generate samples for

    Optional
    --------
    samples : int
        The number of samples to generate for each factor (Default: n)
    criterion : str
        Allowable values are "center" or "c", "maximin" or "m",
        "centermaximin" or "cm", and "correlation" or "corr". If no value
        given, the design is simply randomized.
    iterations : int
        The number of iterations in the maximin and correlations algorithms
        (Default: 5).
    randomstate : np.random.RandomState, int
         Random state (or seed-number) which controls the seed and random draws
    correlation_matrix : ndarray
         Enforce correlation between factors (only used in lhsmu)

    Returns
    -------
    H : 2d-array
        An n-by-samples design matrix that has been normalized so factor values
        are uniformly spaced between zero and one.

    Example
    -------
    A 3-factor design (defaults to 3 samples)::

        >>> lhs(3, random_state=42)
        array([[ 0.12484671,  0.95539205,  0.24399798],
               [ 0.53288616,  0.38533955,  0.86703834],
               [ 0.68602787,  0.31690477,  0.38533151]])

    A 4-factor design with 6 samples::

        >>> lhs(4, samples=6, random_state=42)
        array([[ 0.06242335,  0.19266575,  0.88202411,  0.89439364],
               [ 0.19266977,  0.53538985,  0.53030416,  0.49498498],
               [ 0.71737371,  0.75412607,  0.17634727,  0.71520486],
               [ 0.63874044,  0.85658231,  0.33676408,  0.31102936],
               [ 0.43351917,  0.45134543,  0.12199899,  0.53056742],
               [ 0.93530882,  0.15845238,  0.7386575 ,  0.09977641]])

    A 2-factor design with 5 centered samples::

        >>> lhs(2, samples=5, criterion='center', random_state=42)
        array([[ 0.1,  0.9],
               [ 0.5,  0.5],
               [ 0.7,  0.1],
               [ 0.3,  0.7],
               [ 0.9,  0.3]])

    A 3-factor design with 4 samples where the minimum distance between
    all samples has been maximized::

        >>> lhs(3, samples=4, criterion='maximin', random_state=42)
        array([[ 0.69754389,  0.2997106 ,  0.96250964],
               [ 0.10585037,  0.09872038,  0.73157522],
               [ 0.25351996,  0.65148999,  0.07337204],
               [ 0.91276926,  0.97873992,  0.42783549]])

    A 4-factor design with 5 samples where the samples are as uncorrelated
    as possible (within 10 iterations)::

        >>> lhs(4, samples=5, criterion='correlation', iterations=10, random_state=42)
        array([[ 0.72088348,  0.05121366,  0.97609357,  0.92487081],
               [ 0.49507404,  0.51265511,  0.00808672,  0.37915272],
               [ 0.22217816,  0.2878673 ,  0.24034384,  0.42786629],
               [ 0.91977309,  0.93895699,  0.64061224,  0.14213258],
               [ 0.04719698,  0.70796822,  0.53910322,  0.78857071]])

    """
    H = None

    if random_state is None:
        random_state = np.random.RandomState()
    elif not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)

    if samples is None:
        samples = n

    if criterion is not None:
        if not criterion.lower() in ('center', 'c', 'maximin', 'm',
                                     'centermaximin', 'cm', 'correlation',
                                     'corr','lhsmu'):
            raise ValueError('Invalid value for "criterion": {}'.format(criterion))

    else:
        H = _lhsclassic(n, samples, random_state)

    if criterion is None:
        criterion = 'center'
    if iterations is None:
        iterations = 5

    if H is None:
        if criterion.lower() in ('center', 'c'):
            H = _lhscentered(n, samples, random_state)
        elif criterion.lower() in ('maximin', 'm'):
            H = _lhsmaximin(n, samples, iterations, 'maximin', random_state)
        elif criterion.lower() in ('centermaximin', 'cm'):
            H = _lhsmaximin(n, samples, iterations, 'centermaximin', random_state)
        elif criterion.lower() in ('correlation', 'corr'):
            H = _lhscorrelate(n, samples, iterations, random_state)
        elif criterion.lower() in ('lhsmu'):
            # as specified by the paper. M is set to 5
            H = _lhsmu(n, samples, correlation_matrix, random_state, M=5)

    return H

################################################################################

def _lhsclassic(n, samples, randomstate):
    # Generate the intervals
    cut = np.linspace(0, 1, samples + 1)    
    
    # Fill points uniformly in each interval
    u = randomstate.rand(samples, n)
    a = cut[:samples]
    b = cut[1:samples + 1]
    rdpoints = np.zeros_like(u)
    for j in range(n):
        rdpoints[:, j] = u[:, j]*(b-a) + a
    
    # Make the random pairings
    H = np.zeros_like(rdpoints)
    for j in range(n):
        order = randomstate.permutation(range(samples))
        H[:, j] = rdpoints[order, j]
    
    return H
    
################################################################################

def _lhscentered(n, samples, randomstate):
    # Generate the intervals
    cut = np.linspace(0, 1, samples + 1)    
    
    # Fill points uniformly in each interval
    u = randomstate.rand(samples, n)
    a = cut[:samples]
    b = cut[1:samples + 1]
    _center = (a + b)/2
    
    # Make the random pairings
    H = np.zeros_like(u)
    for j in range(n):
        H[:, j] = randomstate.permutation(_center)
    
    return H
    
################################################################################

def _lhsmaximin(n, samples, iterations, lhstype, randomstate):
    maxdist = 0
    
    # Maximize the minimum distance between points
    for i in range(iterations):
        if lhstype=='maximin':
            Hcandidate = _lhsclassic(n, samples, randomstate)
        else:
            Hcandidate = _lhscentered(n, samples, randomstate)
        
        d = spatial.distance.pdist(Hcandidate, 'euclidean')
        if maxdist<np.min(d):
            maxdist = np.min(d)
            H = Hcandidate.copy()
    
    return H

################################################################################

def _lhscorrelate(n, samples, iterations, randomstate):
    mincorr = np.inf
    
    # Minimize the components correlation coefficients
    for i in range(iterations):
        # Generate a random LHS
        Hcandidate = _lhsclassic(n, samples, randomstate)
        R = np.corrcoef(Hcandidate.T)
        if np.max(np.abs(R[R!=1]))<mincorr:
            mincorr = np.max(np.abs(R-np.eye(R.shape[0])))
            H = Hcandidate.copy()
    
    return H
 
 ################################################################################

def _lhsmu(N, samples=None, corr=None, random_state=None, M=5):

    if random_state is None:
        random_state = np.random.RandomState()
    elif not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)

    if samples is None:
        samples = N

    I = M*samples

    rdpoints = random_state.uniform(size=(I, N))

    dist = spatial.distance.cdist(rdpoints, rdpoints, metric='euclidean')
    D_ij = ma.masked_array(dist, mask=np.identity(I))

    index_rm = np.zeros(I-samples, dtype=int)
    i = 0
    while i < I-samples:
        order = ma.sort(D_ij, axis=1)

        avg_dist = ma.mean(order[:, 0:2], axis=1)
        min_l = ma.argmin(avg_dist)

        D_ij[min_l, :] = ma.masked
        D_ij[:, min_l] = ma.masked

        index_rm[i] = min_l
        i += 1

    rdpoints = np.delete(rdpoints, index_rm, axis=0)

    if(corr is not None):
        #check if covariance matrix is valid
        assert type(corr) == np.ndarray
        assert corr.ndim == 2
        assert corr.shape[0] == corr.shape[1]
        assert corr.shape[0] == N

        norm_u = stats.norm().ppf(rdpoints)
        L = linalg.cholesky(corr, lower=True)

        norm_u = np.matmul(norm_u, L)

        H = stats.norm().cdf(norm_u)
    else:
        H = np.zeros_like(rdpoints, dtype=float)
        rank = np.argsort(rdpoints, axis=0)

        for l in range(samples):
            low = float(l)/samples
            high = float(l+1)/samples

            l_pos = rank == l
            H[l_pos] = random_state.uniform(low, high, size=N)
    return H
