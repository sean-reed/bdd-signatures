from igraph import Graph
from signatures.networks import get_st_bdd_from_cutsets, get_component_types_dict
from signatures.sig_funs import compute_signatures, print_survival_sig_table

if __name__ == "__main__":
    network = Graph.Formula("s -- A:B, A -- C:D, B -- C:E, C -- D:E, D -- t, E -- t")

    for v in network.vs:
        if v["name"] in ["A", "E"]:
            v["compType"] = 1
        elif v["name"] in ["B", "C", "D"]:
            v["compType"] = 2

    bdd, root = get_st_bdd_from_cutsets(network)

    component_types = get_component_types_dict(network)

    sig_dim_types, survival_sig, system_sig = compute_signatures(bdd, root, component_types)
    print_survival_sig_table(survival_sig, sig_dim_types)
