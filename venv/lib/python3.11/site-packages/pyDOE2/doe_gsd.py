"""
Copyright (C) 2018 - Rickard Sjoegren
"""
import itertools

import numpy as np


def gsd(levels, reduction, n=1):
    """
    Create a Generalized Subset Design (GSD).

    Parameters
    ----------
    levels : array-like
        Number of factor levels per factor in design.
    reduction : int
        Reduction factor (bigger than 1). Larger `reduction` means fewer
        experiments in the design and more possible complementary designs.
    n : int
        Number of complementary GSD-designs (default 1). The complementary
        designs are balanced analogous to fold-over in two-level fractional
        factorial designs.

    Returns
    -------
    H : 2d-array | list of 2d-arrays
        `n` m-by-k matrices where k is the number of factors (equal
        to the length of `factor_levels`. The number of rows, m, will
        be approximately equal to the grand product of the factor levels
        divided by `reduction`.

    Raises
    ------
    ValueError
        If input is valid or if design construction fails. Design can fail
        if `reduction` is too large compared to values of `levels`.

    Notes
    -----
    The Generalized Subset Design (GSD) [1]_ or generalized factorial design is
    a generalization of traditional fractional factorial designs to problems
    where factors can have more than two levels.

    In many application problems factors can have categorical or quantitative
    factors on more than two levels. Previous reduced designs have not been
    able to deal with such types of problems. Full multi-level factorial
    designs can handle such problems but are however not economical regarding
    the number of experiments.

    Note for commercial users, the application of GSD to testing of product
    characteristics in a processing facility is patented [2]_

    Examples
    --------
    An example with three factors using three, four and
    six levels respectively reduced with a factor 4 ::

        >>> gsd([3, 4, 6], 4)
        array([[0, 0, 0],
               [0, 0, 4],
               [0, 1, 1],
               [0, 1, 5],
               [0, 2, 2],
               [0, 3, 3],
               [1, 0, 1],
               [1, 0, 5],
               [1, 1, 2],
               [1, 2, 3],
               [1, 3, 0],
               [1, 3, 4],
               [2, 0, 2],
               [2, 1, 3],
               [2, 2, 0],
               [2, 2, 4],
               [2, 3, 1],
               [2, 3, 5]])

    Two complementary designs with two factors using three and
    four levels reduced with a factor 2 ::

        >>> gsd([3, 4], 2, n=2)[0]
        array([[0, 0],
               [0, 2],
               [2, 0],
               [2, 2],
               [1, 1],
               [1, 3]])
        >>> gsd([3, 4], 2, n=2)[1]
        array([[0, 1],
               [0, 3],
               [2, 1],
               [2, 3],
               [1, 0],
               [1, 2]])

    If design fails ValueError is raised ::

        >>> gsd([2, 3], 5)
        Traceback (most recent call last):
         ...
        ValueError: reduction too large compared to factor levels

    References
    ----------
    .. [1] Surowiec, Izabella, Ludvig Vikstrom, Gustaf Hector, Erik Johansson,
       Conny Vikstrom, and Johan Trygg. "Generalized Subset Designs in
       Analytical Chemistry." Analytical Chemistry 89, no. 12 (June 20, 2017):
       6491-97. https://doi.org/10.1021/acs.analchem.7b00506.

    .. [2] Vikstrom, Ludvig, Conny Vikstrom, Erik Johansson, and Gustaf Hector.
       Computer-implemented systems and methods for generating
       generalized fractional designs. US9746850 B2, filed May 9,
       2014, and issued August 29, 2017. http://www.google.se/patents/US9746850.

    """
    try:
        assert all(isinstance(v, int) for v in levels), \
            'levels has to be sequence of integers'
        assert isinstance(reduction, int) and reduction > 1, \
            'reduction has to be integer larger than 1'
        assert isinstance(n, int) and n > 0, \
            'n has to be positive integer'
    except AssertionError as e:
        raise ValueError(e)

    partitions = _make_partitions(levels, reduction)
    latin_square = _make_latin_square(reduction)
    ortogonal_arrays = _make_orthogonal_arrays(latin_square, len(levels))

    try:
        designs = [_map_partitions_to_design(partitions, oa) - 1 for oa in
                   ortogonal_arrays]
    except ValueError:
        raise ValueError('reduction too large compared to factor levels')

    if n == 1:
        return designs[0]
    else:
        return designs[:n]


def _make_orthogonal_arrays(latin_square, n_cols):
    """
    Augment latin-square to the specified number of columns to produce
    an orthogonal array.
    """
    p = len(latin_square)

    first_row = latin_square[0]
    A_matrices = [np.array([[v]]) for v in first_row]

    while A_matrices[0].shape[1] < n_cols:
        new_A_matrices = list()

        for i, A_matrix in enumerate(A_matrices):
            sub_a = list()
            for constant, other_A in zip(first_row,
                                         np.array(A_matrices)[latin_square[i]]):
                constant_vec = np.repeat(constant, len(other_A))[:, np.newaxis]
                combined = np.hstack([constant_vec, other_A])
                sub_a.append(combined)

            new_A_matrices.append(np.vstack(sub_a))

        A_matrices = new_A_matrices

        if A_matrices[0].shape[1] == n_cols:
            break

    return A_matrices


def _map_partitions_to_design(partitions, ortogonal_array):
    """
    Map partitioned factor to final design using orthogonal-array produced
    by augmenting latin square.
    """
    assert len(
        partitions) == ortogonal_array.max() + 1 and ortogonal_array.min() == 0, \
        'Orthogonal array indexing does not match partition structure'

    mappings = list()
    for row in ortogonal_array:
        if any(not partitions[p][factor] for factor, p in enumerate(row)):
            continue

        partition_sets = [partitions[p][factor] for factor, p in enumerate(row)]
        mapping = list(itertools.product(*partition_sets))
        mappings.append(mapping)

    return np.vstack(mappings)


def _make_partitions(factor_levels, num_partitions):
    """
    Balanced partitioning of factors.
    """
    partitions = list()
    for partition_i in range(1, num_partitions + 1):
        partition = list()

        for num_levels in factor_levels:
            part = list()
            for level_i in range(1, num_levels):
                index = partition_i + (level_i - 1) * num_partitions
                if index <= num_levels:
                    part.append(index)

            partition.append(part)

        partitions.append(partition)

    return partitions


def _make_latin_square(n):
    numbers = np.arange(n)
    latin_square = np.vstack([np.roll(numbers, -i) for i in range(n)])
    return latin_square
