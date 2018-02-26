import unittest
import numpy as np
from igraph import Graph
from ..networks import get_st_bdd_from_cutsets, get_component_types_dict
from ..sig_funs import *
from ..trees import *
from collections import OrderedDict


class SignaturesTestCase(unittest.TestCase):
    def test_survival_signature_values_correct_for_networks(self):
        """
        Tests that survival signature is calculated correctly for various network problems.
        """

        # Simple series network with single component type.
        network = Graph.Formula("s -- 1 -- 2 -- t")
        for v in network.vs:
            v["compType"] = 1
        component_types = get_component_types_dict(network)
        bdd, root = get_st_bdd_from_cutsets(network)
        sig_dim_types, survival_sig, system_sig = compute_signatures(bdd, root,
                                                                                             component_types)
        expected_survival_sig = np.array((0, 0, 1))
        expected_system_sig = np.array((1, 0))
        self.assertTrue(np.array_equal(survival_sig, expected_survival_sig))
        self.assertTrue(np.array_equal(system_sig, expected_system_sig))

        '''Network from Figure 1 of "Generalizing the signature to systems with multiple types of components" 
        by F. Coolen and Coolen-Maturi, 2012'''
        network = Graph.Formula("s -- 1, 1 -- 2:3, 2 -- 4:5, 3 -- 4:6, 4 -- 5:6, 5 -- t, 6 -- t")

        for v in (v for v in network.vs if v["name"] in ["1", "2", "5"]): v["compType"] = 1

        for v in (v for v in network.vs if v["name"] in ["3", "4", "6"]): v["compType"] = 2

        bdd, root = get_st_bdd_from_cutsets(network)

        component_types = get_component_types_dict(network)

        sig_dim_types, survival_sig, system_sig = compute_signatures(bdd, root, component_types)

        expected_sig = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.1111, 0.3333],
                                 [0.0, 0.0, 0.4444, 0.6667], [1.0, 1.0, 1.0, 1.0]])

        self.assertTrue(np.array_equal(np.around(survival_sig, decimals=4), expected_sig))

        '''Networks from Figures 2 and Figure 4 of the paper
        "Bayesian inference for reliability of systems and networks using the survival signature" by L. Aslett et
         al, 2015'''
        # Reliability network from Fig. 2 and survival signature result from Table 1.
        network = Graph.Formula("s -- 5:6:7, 5 -- 1, 6 -- 2, 7 -- 3, 1 -- 8, 2 --9, 3 -- 10, 8 -- 4, 9 -- 4,"
                                "10 -- 4, 4 -- 11, 11 -- t")

        for v in (v for v in network.vs if v["name"] in ["1", "2", "3", "4"]): v["compType"] = 1

        for v in (v for v in network.vs if v["name"] in ["5", "6", "7", "8", "9", "10", "11"]):
            v["compType"] = 2

        bdd, root = get_st_bdd_from_cutsets(network)

        component_types = get_component_types_dict(network)

        sig_dim_types, survival_sig, system_sig = compute_signatures(bdd, root, component_types)
        expected_sig = np.zeros((5, 8))
        expected_sig[2, 3] = 0.014
        expected_sig[2, 4] = 0.057
        expected_sig[2, 5] = 0.143
        expected_sig[2, 6] = 0.286
        expected_sig[2, 7] = 0.500
        expected_sig[3, 3] = 0.043
        expected_sig[3, 4] = 0.171
        expected_sig[3, 5] = 0.393
        expected_sig[3, 6] = 0.643
        expected_sig[3, 7] = 0.750
        expected_sig[4, 3] = 0.086
        expected_sig[4, 4] = 0.343
        expected_sig[4, 5] = 0.714
        expected_sig[4, 6] = 0.857
        expected_sig[4, 7] = 1.000

        self.assertTrue(np.array_equal(np.around(survival_sig, decimals=3), expected_sig))

        # Reliability network from Fig. 4 and result from Appendix A.
        network = Graph.Formula("s -- 1:7:9, 1 -- 2:4:5, 9 -- 10, 7 -- 8:10, 2 -- 3, 4 -- 6, 5 -- 6,"
                                "10 -- 8:11, 8 -- t, 3 -- t, 6 -- t, 11 -- t")

        for v in (v for v in network.vs if v["name"] in ["1", "6", "11"]): v["compType"] = 1

        for v in (v for v in network.vs if v["name"] in ["2", "3", "9"]): v["compType"] = 2

        for v in (v for v in network.vs if v["name"] in ["4", "5", "10"]): v["compType"] = 3

        for v in (v for v in network.vs if v["name"] in ["7", "8"]): v["compType"] = 4

        bdd, root = get_st_bdd_from_cutsets(network)

        component_types = get_component_types_dict(network)

        sig_dim_types, survival_sig, system_sig = compute_signatures(bdd, root, component_types)

        # Test a set of selected values from the signature.
        self.assertAlmostEqual(survival_sig[0, 0, 0, 0], 0)
        self.assertAlmostEqual(survival_sig[0, 1, 3, 1], 1.0 / 6)
        self.assertAlmostEqual(survival_sig[1, 1, 2, 0], 2.0 / 27)
        self.assertAlmostEqual(survival_sig[2, 1, 1, 1], 7.0 / 18)
        self.assertAlmostEqual(survival_sig[2, 3, 1, 1], 7.0 / 9)
        self.assertAlmostEqual(survival_sig[3, 2, 0, 1], 1.0 / 3)
        self.assertAlmostEqual(survival_sig[3, 2, 1, 0], 1.0)

    def test_fault_tree__euro3_signature_is_correct(self):
        """Compare top event probability for euro3 fault tree calculated directly from BDD with value calculated from
         signature."""
        gates_filename = "../Data/euro3.txt"
        be_filename = "../Data/euro3_be.txt"
        top_gate_name = "ROOT"
        basic_event_probabilities = {}
        with open(be_filename) as f:
            for line in f.readlines():
                name, prob = line.strip().split(' ')
                basic_event_probabilities[name] = float(prob)

        # Read in gates from file.
        gates = {}
        with open(gates_filename) as f:
            lines = f.readlines()
            while lines:
                parts = lines.pop().strip().split(' ')
                name, op, inputs = parts[0], parts[1], parts[2:]
                gates[name] = (name, op, inputs)

        ordered_bes = get_depth_first_traversal_ordering(top_gate_name, gates, basic_event_probabilities.keys())
        bdd, bdd_root = success_tree_to_bdd(top_gate_name, gates.values(), ordered_bes)
        sig_dim_types, survival_sig, system_sig = compute_signatures(bdd, bdd_root, basic_event_probabilities)

        direct_p = eval_prob(bdd, bdd_root, basic_event_probabilities)
        signature_p = eval_prob_from_surv_sig(survival_sig, sig_dim_types)

        self.assertAlmostEqual(direct_p, signature_p, places=25)


if __name__ == "__main__":
    unittest.main()
