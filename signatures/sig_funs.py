import numpy as np
from math import ceil, floor, pow, factorial
from collections import defaultdict
from itertools import product



def get_element_indices(dimension, dimensions_count, dim_index):
    return tuple((0 if i != dimension else dim_index for i in range(dimensions_count)))


def get_signature_dimension_indices(shape, dimension):
    ndims = len(shape)
    shift_indices = tuple((tuple((slice(0, shape[i]) if i != dimension else dim_index for i in range(0, ndims))) for
                           dim_index in range(0, shape[dimension])))

    return shift_indices


def pad(array, shape, dtype):
    """
    Creates an array of shape 'shape' containing the data from 'array' by padding each dimension with zeros,
     where 'shape' has same number of dimensions as 'array' and size in all dimensions of former is greater than latter.
    :param array: The array from which data is copied.
    :param shape: The shape of the padded array.
    :return: The padded array.
    """
    padded = np.pad(array, list(zip(np.zeros(array.ndim, np.uint8), np.subtract(shape, array.shape))), mode='constant')

    if padded.dtype != dtype:
        padded = padded.astype(dtype, casting="safe")

    return padded


def get_dtype(component_type_counts):
    # Choose appropriate data type for survival signature array based on maximum combinations to be stored.

    max_comb = np.prod([max(nCr(i, ceil(i / 2.0)), nCr(i, floor(i / 2.0))) for i in component_type_counts])

    if max_comb <= 255:
        dt = np.uint8
    elif max_comb <= 65535:
        dt = np.uint16
    elif max_comb <= 4294967295:
        dt = np.uint32
    elif max_comb <= 18446744073709551615:
        dt = np.uint64
    else:
        dt = np.float64

    return dt


def compute_edge_results_for_terminal_one_node(m, K, shape_by_level, dtype_by_level, sig_dim_by_level):
    t1_computed_edge_results = {}

    max_level = m - 1
    t1_0edge = np.ones((1,) * K, dtype_by_level[max_level])

    for current_level in range(max_level, -1, -1):
        t1_0edge = pad(t1_0edge, shape_by_level[current_level], dtype_by_level[current_level])
        t1_1edge = np.roll(t1_0edge, shift=1, axis=sig_dim_by_level[current_level])
        t1_computed_edge_results[(current_level, False)] = t1_0edge
        t1_computed_edge_results[(current_level, True)] = t1_1edge
        t1_0edge = t1_0edge + t1_1edge

    return t1_computed_edge_results


def compute_signatures(bdd, root, var_types=None):
    """
    Computes survival and system signatures from a binary decision diagram (BDD).
    :param bdd: A dd BDD instance with a single root node.
    :param root: The root node of the BDD instance.
    :param var_types (optional): A dictionary mapping each variable in the BDD to its component type
    type. If None then all variables are assumed to be the same type.
    :returns: (sig_dim_types, survival_sig, system_sig)
     where 'sig_dim_types' is a tuple of the component types corresponding to each dimension of the survival signature, 'survival_sig' a NumPy array representing the survival signature,
     and 'system_sig' is a NumPy array representing the system signature or None if there were multiple variable types.
    """

    # If var_types is None then assume all variables are the same component type.
    if var_types is None:
        var_types = defaultdict(int)

    if var_types:
        sig_dim_types = tuple(set(var_types.values()))  # Tuple of var type on each axis of survival signature.
    else:
        sig_dim_types = (0,)

    m = len(bdd.vars)  # Number of variables.

    K = len(sig_dim_types)  # Number of component types.

    #  Build the survival signature shape tuple and tuple of component types for the signature array dimensions.
    var_count_by_type = defaultdict(int)
    shape_by_level = [None] * m
    dtype_by_level = [None] * m

    var_type_counts_by_level = [None] * m

    min_level = 0
    max_level = m - 1

    for level in range(max_level, min_level - 1, -1):
        var = bdd.var_at_level(level)
        var_type = var_types[var]
        var_count_by_type[var_type] += 1
        var_type_counts = tuple((var_count_by_type[t] for t in sig_dim_types))
        var_type_counts_by_level[level] = var_type_counts_by_level
        sig_shape = tuple((var_count_by_type[t] + 1 for t in sig_dim_types))
        shape_by_level[level] = sig_shape

        # Get data type to store maximum count.
        sig_dt = get_dtype(var_type_counts)
        dtype_by_level[level] = sig_dt

    # Compute and store tuple of indices for each dimension of the signature array, where tuple element i for
    # dimension j selects all elements at index of i of dimension j of the signature array.
    sig_dimension_indices_by_dimension = []
    for dim in range(0, len(sig_shape)):
        sig_dimension_indices_by_dimension.append(get_signature_dimension_indices(sig_shape, dim))

    # Build dictionaries mapping (a) BDD levels to a tuple with the dimension of the corresponding component type and
    # (b) BDD levels to indices for that dimension in the signature table.
    sig_dim_by_level = np.empty(len(var_types), dtype=np.uint8)
    for var in var_types:
        var_level = bdd.level_of_var(var)
        var_type = var_types[var]
        var_type_sig_dim = sig_dim_types.index(var_type)
        sig_dim_by_level[var_level] = var_type_sig_dim

    # Compute signature array for terminal one node at each level to account for missing vars in BDD.
    t1_computed_edge_results = compute_edge_results_for_terminal_one_node(m, K, shape_by_level,
                                                                          dtype_by_level, sig_dim_by_level)

    sig_norm = t1_computed_edge_results[(0, False)] + t1_computed_edge_results[(0, True)]

    ite_computed = {}  # Key: node, Value: (level, signature array).
    if root == bdd.true:
        st = sig_norm
    elif root == bdd.false:
        st = 0
    else:
        level, st = rec_calculate_signature(bdd, root, sig_dim_by_level, ite_computed,
                                            t1_computed_edge_results, shape_by_level, dtype_by_level)

        # Adjust for missing variables at higher levels than root in the bdd.
        st = pad(st, shape_by_level[0], dtype_by_level[0])
        for i in range(level - 1, -1, -1):
            st += np.roll(st, shift=1, axis=sig_dim_by_level[i])

    # Normalise the signature array for the root BDD node.
    survival_sig = np.true_divide(st, sig_norm)

    # If there is only a single component type then also calculate the system signature.
    if len(sig_shape) == 1:
        system_sig = np.flipud(np.ediff1d(survival_sig))
    else:
        system_sig = None

    return sig_dim_types, survival_sig, system_sig


def rec_calculate_signature(bdd, node, sig_dim_by_level, ite_computed, t1_computed_edge_results,
                            shape_by_level, dtype_by_level):
    if node < 0:
        level, low, high = bdd._succ[-node]
        low = -low
        high = -high
    else:
        level, low, high = bdd._succ[node]

    node_axis = sig_dim_by_level[level]

    # Get signature array from low descendant.
    if low == bdd.true:
        low_st = t1_computed_edge_results[(level, False)]
    elif low == bdd.false:
        low_st = 0
    else:  # ite node - at higher level so must be computed already.
        low_level, low_st = ite_computed.get(low, (None, None))
        if low_st is None:
            low_level, low_st = \
                rec_calculate_signature(bdd, low, sig_dim_by_level, ite_computed,
                                        t1_computed_edge_results, shape_by_level, dtype_by_level)
        # Adjust signature array for missing (doesn't matter) vars in BDD.
        low_st = pad(low_st, shape_by_level[level], dtype_by_level[level])
        # update_st_for_missing_vars(low_st, low_level, level, sig_dim_by_level, st_dimension_indices_by_level)
        for i in range(low_level - 1, level, -1):
            low_st += np.roll(low_st, shift=1, axis=sig_dim_by_level[i])

    # Get signature array from high descendant.
    if high == bdd.true:
        high_st = t1_computed_edge_results[(level, True)]
    elif high == bdd.false:
        high_st = 0
    else:  # ite node - at higher level so must be computed already.
        high_level, high_st = ite_computed.get(high, (None, None))
        if high_st is None:
            high_level, high_st = \
                rec_calculate_signature(bdd, high, sig_dim_by_level, ite_computed,
                                        t1_computed_edge_results, shape_by_level, dtype_by_level)
        high_st = pad(high_st, shape_by_level[level], dtype_by_level[level])
        # update_st_for_missing_vars(high_st, high_level, level, sig_dim_by_level, st_dimension_indices_by_level)
        for i in range(high_level - 1, level, -1):
            high_st += np.roll(high_st, shift=1, axis=sig_dim_by_level[i])

        # Update signature array to add survival of this node's component.
        high_st = np.roll(high_st, shift=1, axis=node_axis)

    # Calculate signature array for this node as sum of low and high descendants and store result in cache.
    st = low_st + high_st
    result = (level, st)
    ite_computed[node] = result

    return result


def print_system_sig_table(signature):
    """
    Prints a formatted table of a system signature.
    :param signature: The system signature to be printed.
    """
    print("Number of Failures | Probability")
    print("")
    for i in range(len(signature)):
        print("{:18} | {:.6f}".format(i+1, signature[i]))


def print_survival_sig_table(signature, sig_dim_types):
    """
    Prints a formatted table of a survival signature.
    :param signature: The survival signature to be printed.
    :param sig_dim_types: The component types for the dimensions in the signature array.
    """
    print(" | ".join(["Type {:3}".format(t) for t in sig_dim_types]) + " | Probability")
    print("")
    var_type_counts = (s - 1 for s in signature.shape)
    for indices in product(*[range(0, count + 1) for count in var_type_counts]):
        print(" | ".join(["{:8}".format(index) for index in indices]) + " | {:.6f}".format(signature[indices]))


def eval_prob(bdd, node, var_probabilities, computed={}):
    if node == bdd.true:
        result = 1.0
    elif node == bdd.false:
        result = 0.0
    else:
        result = computed.get(node, None)
        if result is not None:
            return result

        if node < 0:
            level, u, v = bdd.succ[-node]
            u_prob = eval_prob(bdd, -u, var_probabilities, computed)
            v_prob = eval_prob(bdd, -v, var_probabilities, computed)
        else:
            level, u, v = bdd._succ[node]
            u_prob = eval_prob(bdd, u, var_probabilities, computed)
            v_prob = eval_prob(bdd, v, var_probabilities, computed)

        var = bdd.var_at_level(level)
        var_prob = var_probabilities[var]
        result = ((1.0 - var_prob) * u_prob) + (var_prob * v_prob)

        computed[node] = result

    return result

def eval_prob_from_surv_sig(signature, sig_dim_probabilities):
    """
    Computes the survival probability of a system from its signature and probability of survival for components of each
    type.
    :param signature: The survival signature.
    :param sig_dim_probabilities: A tuple giving the probability of survival for the component type corresponding to
     each dimension of the signature.
    :return: The probability the system survives.
    """
    component_type_counts = [i - 1 for i in signature.shape]
    sig_probabilities = \
        np.fromfunction(np.vectorize(lambda *indices: np.prod([nCr(n, k) * (pow(prob, k) * pow(1.0 - prob, n - k))
                                                               for (prob, n, k) in
                                                               zip(sig_dim_probabilities, component_type_counts,
                                                                   indices)])),
                        signature.shape)

    p = np.sum(signature * sig_probabilities)

    return p


def nCr(n,k):
    """Number of combinations of n things taken k at a time."""
    return factorial(n) // factorial(k) // factorial(n-k)


