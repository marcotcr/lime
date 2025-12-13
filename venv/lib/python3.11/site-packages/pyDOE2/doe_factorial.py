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

import imp
import itertools
import math
import re
import string
from itertools import dropwhile, combinations, islice

import numpy as np
from scipy.special import binom


__all__ = ['np', 'fullfact', 'ff2n', 'fracfact', 'fracfact_by_res', 'fracfact_opt',
    'fracfact_aliasing', 'alias_vector_indices']

def fullfact(levels):
    """
    Create a general full-factorial design
    
    Parameters
    ----------
    levels : array-like
        An array of integers that indicate the number of levels of each input
        design factor.
    
    Returns
    -------
    mat : 2d-array
        The design matrix with coded levels 0 to k-1 for a k-level factor
    
    Example
    -------
    ::
    
        >>> fullfact([2, 4, 3])
        array([[ 0.,  0.,  0.],
               [ 1.,  0.,  0.],
               [ 0.,  1.,  0.],
               [ 1.,  1.,  0.],
               [ 0.,  2.,  0.],
               [ 1.,  2.,  0.],
               [ 0.,  3.,  0.],
               [ 1.,  3.,  0.],
               [ 0.,  0.,  1.],
               [ 1.,  0.,  1.],
               [ 0.,  1.,  1.],
               [ 1.,  1.,  1.],
               [ 0.,  2.,  1.],
               [ 1.,  2.,  1.],
               [ 0.,  3.,  1.],
               [ 1.,  3.,  1.],
               [ 0.,  0.,  2.],
               [ 1.,  0.,  2.],
               [ 0.,  1.,  2.],
               [ 1.,  1.,  2.],
               [ 0.,  2.,  2.],
               [ 1.,  2.,  2.],
               [ 0.,  3.,  2.],
               [ 1.,  3.,  2.]])
               
    """
    n = len(levels)  # number of factors
    nb_lines = np.prod(levels)  # number of trial conditions
    H = np.zeros((nb_lines, n))
    
    level_repeat = 1
    range_repeat = np.prod(levels)
    for i in range(n):
        range_repeat //= levels[i]
        lvl = []
        for j in range(levels[i]):
            lvl += [j]*level_repeat
        rng = lvl*range_repeat
        level_repeat *= levels[i]
        H[:, i] = rng
     
    return H
    
################################################################################

def ff2n(n):
    """
    Create a 2-Level full-factorial design
    
    Parameters
    ----------
    n : int
        The number of factors in the design.
    
    Returns
    -------
    mat : 2d-array
        The design matrix with coded levels -1 and 1
    
    Example
    -------
    ::

        >>> ff2n(3)
        array([[-1., -1., -1.],
               [ 1., -1., -1.],
               [-1.,  1., -1.],
               [ 1.,  1., -1.],
               [-1., -1.,  1.],
               [ 1., -1.,  1.],
               [-1.,  1.,  1.],
               [ 1.,  1.,  1.]])

    """
    return 2*fullfact([2]*n) - 1

################################################################################

def fracfact(gen):
    """
    Create a 2-level fractional-factorial design with a generator string.

    Parameters
    ----------
    gen : str
        A string, consisting of lowercase, uppercase letters or operators "-"
        and "+", indicating the factors of the experiment

    Returns
    -------
    H : 2d-array
        A m-by-n matrix, the fractional factorial design. m is 2^k, where k
        is the number of letters in ``gen``, and n is the total number of
        entries in ``gen``.

    Notes
    -----
    In ``gen`` we define the main factors of the experiment and the factors
    whose levels are the products of the main factors. For example, if

        gen = "a b ab"

    then "a" and "b" are the main factors, while the 3rd factor is the product
    of the first two. If we input uppercase letters in ``gen``, we get the same
    result. We can also use the operators "+" and "-" in ``gen``.

    For example, if

        gen = "a b -ab"

    then the 3rd factor is the opposite of the product of "a" and "b".

    The output matrix includes the two level full factorial design, built by
    the main factors of ``gen``, and the products of the main factors. The
    columns of ``H`` follow the sequence of ``gen``.

    For example, if

        gen = "a b ab c"

    then columns H[:, 0], H[:, 1], and H[:, 3] include the two level full
    factorial design and H[:, 2] includes the products of the main factors.

    Examples
    --------
    ::

        >>> fracfact("a b ab")
        array([[-1., -1.,  1.],
               [ 1., -1., -1.],
               [-1.,  1., -1.],
               [ 1.,  1.,  1.]])

        >>> fracfact("A B AB")
        array([[-1., -1.,  1.],
               [ 1., -1., -1.],
               [-1.,  1., -1.],
               [ 1.,  1.,  1.]])

        >>> fracfact("a b -ab c +abc")
        array([[-1., -1., -1., -1., -1.],
               [ 1., -1.,  1., -1.,  1.],
               [-1.,  1.,  1., -1.,  1.],
               [ 1.,  1., -1., -1., -1.],
               [-1., -1., -1.,  1.,  1.],
               [ 1., -1.,  1.,  1., -1.],
               [-1.,  1.,  1.,  1., -1.],
               [ 1.,  1., -1.,  1.,  1.]])

    """
    # Recognize letters and combinations
    A = [item for item in re.split('\-|\s|\+', gen) if item]  # remove empty strings
    C = [len(item) for item in A]

    # Indices of single letters (main factors)
    I = [i for i, item in enumerate(C) if item==1]

    # Indices of letter combinations (we need them to fill out H2 properly).
    J = [i for i, item in enumerate(C) if item!=1]

    # Check if there are "-" or "+" operators in gen
    U = [item for item in gen.split(' ') if item]  # remove empty strings

    # If R1 is either None or not, the result is not changed, since it is a
    # multiplication of 1.
    R1 = _grep(U, '+')
    R2 = _grep(U, '-')

    # Fill in design with two level factorial design
    H1 = ff2n(len(I))
    H = np.zeros((H1.shape[0], len(C)))
    H[:, I] = H1

    # Recognize combinations and fill in the rest of matrix H2 with the proper
    # products
    for k in J:
        # For lowercase letters
        xx = np.array([ord(c) for c in A[k]]) - 97

        # For uppercase letters
        if np.any(xx<0):
            xx = np.array([ord(c) for c in A[k]]) - 65

        H[:, k] = np.prod(H1[:, xx], axis=1)

    # Update design if gen includes "-" operator
    if R2:
        H[:, R2] *= -1

    # Return the fractional factorial design
    return H


def fracfact_by_res(n, res):
    """
    Create a 2-level fractional factorial design with `n` factors
    and resolution `res`.

    Parameters
    ----------
    n : int
        The number of factors in the design.
    res : int
        Desired design resolution

    Returns
    -------
    H : 2d-array
        A m-by-`n` matrix, the fractional factorial design. m is the
        minimal amount of rows possible for creating a fractional
        factorial design matrix at resolution `res`

    Raises
    ------
    ValueError
        If the current design is not possible to construct.

    Notes
    -----
    The resolution of a design is defined as the length of the shortest
    word in the defining relation. The resolution describes the level of
    confounding between factors and interaction effects, where higher
    resolution indicates lower degree of confounding.

    For example, consider the 2^4-1-design defined by

        gen = "a b c ab"

    The factor "d" is defined by "ab" with defining relation I="abd", where
    I is the unit vector. In this simple example the shortest word is "abd"
    meaning that this is a resolution III-design.

    In practice resolution III-, IV- and V-designs are most commonly applied.

    * III: Main effects may be confounded with two-factor interactions.
    * IV: Main effects are unconfounded by two-factor interactions, but
          two-factor interactions may be confounded with each other.
    * V: Main effects unconfounded with up to four-factor interactions,
         two-factor interactions unconfounded with up to three-factor
         interactions. Three-factor interactions may be confounded with
         each other.

    Examples
    --------
    ::
        >>> fracfact_by_res(6, 3)
        array([[-1., -1., -1.,  1.,  1.,  1.],
               [ 1., -1., -1., -1., -1.,  1.],
               [-1.,  1., -1., -1.,  1., -1.],
               [ 1.,  1., -1.,  1., -1., -1.],
               [-1., -1.,  1.,  1., -1., -1.],
               [ 1., -1.,  1., -1.,  1., -1.],
               [-1.,  1.,  1., -1., -1.,  1.],
               [ 1.,  1.,  1.,  1.,  1.,  1.]])

        >>> fracfact_by_res(5, 5)
        Traceback (most recent call last):
        ...
        ValueError: design not possible
    """
    # Determine minimum required number of base-factors.
    min_fac = next(dropwhile(lambda n_: _n_fac_at_res(n_, res) < n,
                             range(res - 1, n)), None)

    if min_fac is None:
        raise ValueError('design not possible')
    elif min_fac > len(string.ascii_lowercase):
        # This check needs to be done to make sure that the number
        # of available are enough since `fracfact` parses design generator
        # characters. In practice, this is highly theoretical and it is
        # much more likely to run into memory-issues.
        raise ValueError('design requires too many base-factors.')

    # Get base factors.
    factors = list(string.ascii_lowercase[:min_fac])

    # Fill out with factor combinations until `n` factors.
    factor_combs = (''.join(c) for r in range(res - 1, len(factors))
                    for c in combinations(factors, r))
    extra_factors = list(islice(factor_combs, n - len(factors)))

    # Concatenate `gen` string for `fracfact`.
    gen = ' '.join(factors + extra_factors)
    return fracfact(gen)


def _grep(haystack, needle):
    try:
        haystack[0]
    except (TypeError, AttributeError):
        return [0] if needle in haystack else []
    else:
        locs = []
        for idx, item in enumerate(haystack):
            if needle in item:
                locs += [idx]
        return locs

def _n_fac_at_res(n, res):
    """ Calculate number of possible factors for fractional factorial
    design with `n` base factors at resolution `res`.
    """
    return sum(binom(n, r) for r in range(res - 1, n)) + n

################################################################################


def fracfact_opt(n_factors, n_erased, max_attempts=0) :
    """
    Find the optimal generator string for a 2-level fractional-factorial design
    with the specified number of factors and erased factors.

    Parameters
    ----------
    n_factors : int
        The number of factors in the full factorial design
    n_erased : int
        The number of factors to "remove" to create the fractional design
    max_attempts : int
        The design is searched by exhaustive search, with the most "promising"
        combinations attempted first. For large designs it might be unfeasible
        to attempt all combinations.
        Posite values give the number of models to attemps. Zero or negative
        values indicate all combinations should be attempted.

    Returns
    -------
    gen : str
        A generator string in the format expected by fracfact() with the 2^k-p
        design, where k=n_factors and p=n_erased. The design disallows aliasing
        of main factors, and minimizes aliasing of low-order interactions.
    alias_map : list of str
        The map of aliases that the design inflicts.
        More details in fracfact_aliasing().
    alias_vector : 1d numpy.array
        The vector with the cost of the design in term of aliasings.
        More details in fracfact_aliasing().
    """
    def n_comb(n, k):
        if k<=0 or n<=0 or k>n: return 0
        return math.factorial(n) / (math.factorial(k) * math.factorial(n-k))

    if n_factors > 20:
        raise ValueError('Design too big, use 20 factors or less')

    if n_factors<2:
        raise ValueError('Design too small')

    if n_erased<0:
        raise ValueError('Number of erased factors must be non-negative')

    n_main_factors = n_factors-n_erased
    n_aliases = sum(( n_comb(n_main_factors, n)
                      for n in range(2, n_main_factors+1) ))

    if n_erased>n_comb(n_aliases, n_erased):
        raise ValueError('Too many erased factors to create aliasing')

    all_names = string.ascii_lowercase
    factors = range(n_factors)
    main_factors = range(n_main_factors)
    main_design = ' '.join([ all_names[f] for f in main_factors ])
    aliases = itertools.chain.from_iterable((
            itertools.combinations(main_factors, n)
            for n in range(2, n_main_factors+1) ))

    aliases = sorted(list(aliases), key=lambda a : (len(a), a), reverse=True)
    best_design = None
    best_map = []
    best_vector = np.repeat(n_factors, n_factors)
    design_shape = (2**n_main_factors, n_factors)
    all_combinations = itertools.combinations(aliases, n_erased)
    all_combinations = (all_combinations if max_attempts<=0 else
            itertools.islice(all_combinations, 0, max_attempts))

    for aliasing in all_combinations :
        aliasing_design = ' '.join([
                ''.join([ all_names[f] for f in a ]) for a in aliasing ])
        complete_design = main_design+' '+aliasing_design
        design = fracfact(complete_design)
        assert design.shape == design_shape
        alias_map, alias_vector = fracfact_aliasing(design)
        if list(alias_vector) < list(best_vector):
            best_design = complete_design
            best_map = alias_map
            best_vector = alias_vector

    return best_design, best_map, best_vector


def fracfact_aliasing(design):
    """
    Find the aliasings in a design, given the contrasts.

    Parameters
    ----------
    design : numpy 2d array
        A design like those returned by fracfact()

    Returns
    -------
    alias_map : list of str
        The map of aliases that the design inflicts. Each string in the list is
        a set of factors and interactions that are aliased among themselves, in
        the format a = bcd = def etc. If there is no aliasing (n_erased=0) the
        map simply lists all factors and interactions.
    alias_vector : 1d numpy.array
        The vector with the cost of the design in term of aliasings. Each cell
        in the array counts the number of aliasing between factors/interactions
        of size i and of size j, as given by alias_vector_indices().

        The alias cost vector can be turned into a more explicit upper
        triangular cost matrix with the idiom:
        alias_matrix = np.zeros((n_factors, n_factors,))
        alias_matrix[alias_vector_indices(n_factors)] = alias_vector

        The entry in alias_matrix[i,j] (i<=j) shows how many aliasings where
        created among i-th order interactions and j-th order interactions.
    """
    n_rounds, n_factors = design.shape

    if n_factors > 20:
        raise ValueError('Design too big, use 20 factors or less')

    all_names = string.ascii_lowercase
    factors = range(n_factors)
    all_combinations = itertools.chain.from_iterable((
            itertools.combinations(factors, n) for n in range(1, n_factors+1) ))
    aliases = {}

    for combination in all_combinations:
        contrast = np.prod(design[:,combination], axis=1)
        contrast.flags.writeable = False
        aliases[contrast.data] = aliases.get(contrast.data, [])
        aliases[contrast.data].append(combination)

    aliases_list = []
    for alias in aliases.values():
        aliases_list.append(sorted(alias, key=lambda a : (len(a), a)))
    aliases_list = sorted(aliases_list,
            key=lambda list : ([len(a) for a in list], list))

    aliases_readable = []
    alias_matrix = np.zeros((n_factors, n_factors, ))

    for alias in aliases_list:
        alias_readable = ' = '.join([
                ''.join([ all_names[f] for f in a ]) for a in alias ])
        aliases_readable.append(alias_readable)
        for sizes in itertools.combinations([ len(a) for a in alias], 2):
            assert sizes[0]>=0 and sizes[1]>=0
            assert sizes[0] <= sizes[1]
            alias_matrix[sizes[0]-1, sizes[1]-1] += 1

    alias_vector = alias_matrix[alias_vector_indices(n_factors)]

    return aliases_readable, alias_vector


def alias_vector_indices(n_factors):
    """
    Find the indexes to convert the alias_vector into a square matrix and
    vice-versa.

    Parameters
    ----------
    n_factors : int
        The number of factors in the full factorial design

    Returns
    -------
    rows : 1d numpy array
    cols : 1d numpy.array
        Rows and columns of the indices of the upper triangular square matrix
        with n_factor rows/columns. This function returns a different indice
        order than numpy.triu_indices, as it puts the indices representing the
        most serious aliasings first, to help in the optimization procedure.
    """
    if n_factors > 20:
        raise ValueError('Design too big, use 20 factors or less')

    indices = list(itertools.combinations_with_replacement(range(n_factors), 2))
    indices = sorted(indices, key=lambda i: max(i))

    rows = np.asarray([ i[0] for i in indices ])
    cols = np.asarray([ i[1] for i in indices ])

    return rows, cols
