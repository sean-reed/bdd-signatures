# About
 A Python 3 implementation of the algorithm for calculating the system and survival signature from a binary decision diagram representation of a system's reliability structure function that was described in the paper  [ "An efficient algorithm for exact computation of system and survival signatures using binary decision diagrams"](https://doi.org/10.1016/j.ress.2017.03.036).
## Getting Started

### Prerequisites
This software depends on the following libraries and their dependencies: [NumPy](http://www.numpy.org/) (for multidimensional arrays), [dd](https://pypi.org/project/dd) (for Binary Decision Diagrams) and [python-igraph](http://igraph.org/python/) (for networks).

You can install them using pip:

```
pip install requirements.txt
```
Note: Windows users should follow the [instructions here](http://igraph.org/python/) for installing the python-igraph library.
### Overview
The compute_signatures function implements the algorithm that computes a multidimensional array representing the survival signature (and also the system signature if there is only a single component type) for a BDD where each variable is mapped to a type.
#### Example
```python
from dd import BDD
from signatures import sig_funs

# Create a simple BDD.
bdd = BDD()
bdd.declare('w','x','y','z')
w = bdd.var('w')
x = bdd.var('x')
y = bdd.var('y')
z = bdd.var('z')
t = bdd.apply('and', w, x)
u = bdd.apply('and', y, z)
v = bdd.apply('or', t, u)

# Create a mapping between the variables and their component types.
var_types = {'w':0, 'x':1, 'y':1, 'z':0}

# Compute signatures for the root node, v, of the BDD.
sig_dim_types, survival_sig, system_sig = sig_funs.compute_signatures(bdd, v, var_types)

# Print the survival signature as a formatted table.
sig_funs.print_survival_sig_table(survival_sig, sig_dim_types)
'''
Type   0 | Type   1 | Probability

       0 |        0 | 0.000000
       0 |        1 | 0.000000
       0 |        2 | 0.000000
       1 |        0 | 0.000000
       1 |        1 | 0.500000
       1 |        2 | 1.000000
       2 |        0 | 0.000000
       2 |        1 | 1.000000
       2 |        2 | 1.000000
'''

# Repeat calculation but with all variables of same component type.
sig_dim_types, survival_sig, system_sig = sig_funs.compute_signatures(bdd, v)

# Print the system signature as a formatted table.
sig_funs.print_system_sig_table(system_sig)
'''
Number of Failures | Probability

                 1 | 0.000000
                 2 | 0.666667
                 3 | 0.333333
                 4 | 0.000000
'''

# Get the system survival probability from the survival signature.
sig_dim_probabilities = (0.85, 0.98)
sys_survival_prob = sig_funs.eval_prob_from_surv_sig(survival_sig, sig_dim_probabilities)
print("Probability: {:.6f}".format(sys_survival_prob))
'''
Probability: 0.0.972111
'''
```
#### Networks
Some helper functions are included for computing signatures for the connectivity between two terminal nodes in a network with unreliable vertices. These are implemented to be familiar to users of the ReliabilityTheory R package and uses the same library (igraph) and notation.

Note that the helper function for constructing a BDD of a network relies on derivation of cut-sets and is computationally intensive for anything except small networks.
#####Example
The following example computes and prints the survival signature for the network from Figure 1 of the paper ["Generalizing the signature to systems with multiple types of components"](https://doi.org/10.1007/978-3-642-30662-4_8) (Coolen and Coolen-Maturi, 2012):
```python
from igraph import Graph
from signatures.networks import get_component_types_dict, get_st_bdd_from_cutsets
from signatures.sig_funs import compute_signatures, print_survival_sig_table

# Create the network.
network = Graph.Formula("s -- A, A -- B:C, B -- D:E, C -- D:F, D -- E:F, E -- t, F -- t")

# Set the component types of the vertices.
for v in network.vs:
  if v["name"] in ["A", "B", "E"]:
    v["compType"] = 1
  elif v["name"] in ["C", "D", "F"]:
    v["compType"] = 2

# Get the binary decision diagram representing the Boolean structure function for connectivity of the 's' and 't' nodes in
# terms of the vertex states.
bdd, root = get_st_bdd_from_cutsets(network)

# Compute the signature of the binary decision diagram.
component_types = get_component_types_dict(network)
sig_dim_types, survival_sig, system_sig = compute_signatures(bdd, root, component_types)

# Print the survival signature as a formatted table.
print_survival_sig_table(survival_sig, sig_dim_types)
'''
Type   1 | Type   2 | Probability

       0 |        0 | 0.000000
       0 |        1 | 0.000000
       0 |        2 | 0.000000
       0 |        3 | 0.000000
       1 |        0 | 0.000000
       1 |        1 | 0.000000
       1 |        2 | 0.111111
       1 |        3 | 0.333333
       2 |        0 | 0.000000
       2 |        1 | 0.000000
       2 |        2 | 0.444444
       2 |        3 | 0.666667
       3 |        0 | 1.000000
       3 |        1 | 1.000000
       3 |        2 | 1.000000
       3 |        3 | 1.000000
'''
```
#### Fault Trees and Success Trees
The trees module contains some functions for computing the BDD for the structure function represented by a [fault tree](https://en.wikipedia.org/wiki/Fault_tree_analysis) or success tree (complement of a fault tree). AND, OR and n-out-of-m gates are supported.
#### Example
```python
from signatures.trees import fault_tree_to_bdd
from signatures.sig_funs import compute_signatures, print_survival_sig_table, print_system_sig_table


# Each basic event is specified as a unique name (amongst all basic events and gates).
basic_events = ["BE1", "BE2", "BE3", "BE4"]

'''
Each gate is specified as a (name, logical_op, inputs) tuple, where:
name - unique identifier string amongst all gates and basic events.
logical_op - "*" to signify an AND gate, "+" to signify an OR gate, "(n/m)" 
to signify an n-out-of-m gate (where n and m are integers).
inputs - tuple of the names of the child gates and basic events.
'''
top_gate_name = "System Event" # System failure (survives) for fault (success) tree
top_gate = (top_gate_name, "*", ("G1", "G2"))
G1 = ("G1", "+", ("BE1", "BE2"))
G2 = ("G2", "+", ("BE3", "BE4"))
gates = [top_gate, G1, G2]

component_types = {"BE1": 1, "BE2": 2, "BE3": 2, "BE4": 1}

# Convert tree to BDD.
bdd, root = fault_tree_to_bdd(top_gate_name, gates, basic_events)
# bdd, root = success_tree_to_bdd(top_gate_name, gates, basic_events)

# Compute signatures and print as table.
sig_dim_types, survival_sig, system_sig = compute_signatures(bdd, root, component_types)
print_survival_sig_table(survival_sig, sig_dim_types)
'''
Type   1 | Type   2 | Probability

       0 |        0 | 0.000000
       0 |        1 | 0.000000
       0 |        2 | 0.000000
       1 |        0 | 0.000000
       1 |        1 | 0.500000
       1 |        2 | 1.000000
       2 |        0 | 0.000000
       2 |        1 | 1.000000
       2 |        2 | 1.000000
'''
```
## Tests
To run the unit tests:
```
python -m unittest
```
## Author
Sean Reed.
## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
## Citation
Please use the following citation for this algorithm:

Reed, S. "An efficient algorithm for exact computation of system and survival signatures using binary decision diagrams",
Reliability Engineering & System Safety, Volume 165, 2017, [https://doi.org/10.1016/j.ress.2017.03.036](https://doi.org/10.1016/j.ress.2017.03.036).

Thank you.