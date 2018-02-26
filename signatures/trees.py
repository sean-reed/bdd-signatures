from dd import BDD
from itertools import combinations
import queue

def get_depth_first_traversal_ordering(top_gate_name, gates, basic_events):
    '''

    :param top_gate_name: The name of the top gate.
    :param gates: List of the gates in the tree.
    :param basic_events: List of the basic events in the tree.
    :return: A list of the basic events in order of a depth first traversal of the tree starting at the top gate.
    '''
    ordering = rec_get_depth_first_traversal_ordering(top_gate_name, gates, basic_events)

    # Add any basic events missing from the fault tree to the start of the ordering array.
    for basic_event in basic_events:
        if basic_event not in ordering:
            ordering.insert(0, basic_event)

    return ordering


def rec_get_depth_first_traversal_ordering(node, gates, basic_events, level=0, ordering=[]):
    if node in basic_events:
        if node not in ordering:
            ordering.append(node)
    else:
        level += 1
        for input in gates[node][2]:
            rec_get_depth_first_traversal_ordering(input, gates, basic_events, level, ordering)

    return ordering


def fault_tree_to_bdd(top_gate_name, gates, basic_events):
    """
    :param top_gate_name: The name of the top gate of the tree.
    :param gates: A list of the gates in the tree.
    :param basic_events: A list of the basic events in the tree. BDD variable order will match the order of the basic
    events in this list.
    :return: A BDD representing the structure function represented by the fault tree rooted at the top gate.
    """
    _bdd = BDD()

    bdd_nodes = {}
    tree_node_names = set()
    # Add basic events from tree to BDD.
    for be in basic_events:
        if be in tree_node_names:
            raise Exception("Basic event name " + be + " not unique.")
        _bdd.add_var(be)
        bdd_nodes[be] = _bdd.var(be)
        tree_node_names.add(be)

    # Build BDD node for each gate in the tree.
    gates_to_parse = list(gates)
    while gates_to_parse:
        gate = gates_to_parse.pop(0)
        name, logical_op, inputs = gate
        if all((input_name in tree_node_names for input_name in inputs)):
            if name in tree_node_names:
                raise Exception("Gate name " + name + " not unique.")
            tree_node_names.add(name)
            if logical_op == "*":  # AND gate.
                result = _bdd.false
                for i in inputs:
                    result = _bdd.apply('or', result, bdd_nodes[i])
            elif logical_op == "+":  # OR gate.
                result = _bdd.true
                for i in inputs:
                    result = _bdd.apply('and', result, bdd_nodes[i])
            else:  # n-out-of-k gate.
                n = logical_op.split('/')[0].strip('(')
                result = _bdd.true
                for comb in combinations([bdd_nodes[i] for i in inputs], int(n)):
                    comb_result = _bdd.false
                    for node in comb:
                        comb_result = _bdd.apply('or', comb_result, node)
                    result = _bdd.apply('and', result, comb_result)

            bdd_nodes[name] = result
        else:
            gates_to_parse.append(gate)

    bdd_root = bdd_nodes[top_gate_name]
    _bdd.incref(bdd_root)
    _bdd.collect_garbage()

    return _bdd, bdd_root


def success_tree_to_bdd(top_gate_name, gates, basic_events):
    """
    :param top_gate_name: The name of the top gate of the tree.
    :param gates: A list of the gates in the tree.
    :param basic_events: A list of the basic events in the tree. BDD variable order will match the order of the basic
    events in this list.
    :return: A BDD representing the structure function represented by the success tree rooted at the top gate.
    """
    _bdd = BDD()

    bdd_nodes = {}
    tree_node_names = set()
    # Add basic events from tree to BDD.
    for be in basic_events:
        if be in tree_node_names:
            raise Exception("Basic event name " + be + " not unique.")
        _bdd.add_var(be)
        bdd_nodes[be] = _bdd.var(be)
        tree_node_names.add(be)

    # Build BDD node for each gate in the tree.
    gates_to_parse = list(gates)
    while gates_to_parse:
        gate = gates_to_parse.pop(0)
        name, logical_op, inputs = gate
        if all((input_name in tree_node_names for input_name in inputs)):
            if name in tree_node_names:
                raise Exception("Gate name " + name + " not unique.")
            tree_node_names.add(name)
            if logical_op == "*":  # AND gate.
                result = _bdd.true
                for i in inputs:
                    result = _bdd.apply('and', result, bdd_nodes[i])
            elif logical_op == "+":  # OR gate.
                result = _bdd.false
                for i in inputs:
                    result = _bdd.apply('or', result, bdd_nodes[i])
            else:  # n-out-of-k gate.
                n = logical_op.split('/')[0].strip('(')
                result = _bdd.false
                for comb in combinations([bdd_nodes[i] for i in inputs], int(n)):
                    comb_result = _bdd.true
                    for node in comb:
                        comb_result = _bdd.apply('and', comb_result, node)
                    result = _bdd.apply('or', result, comb_result)

            bdd_nodes[name] = result
        else:
            gates_to_parse.append(gate)

    bdd_root = bdd_nodes[top_gate_name]
    _bdd.incref(bdd_root)
    _bdd.collect_garbage()

    return _bdd, bdd_root