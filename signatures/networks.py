from dd import autoref as _bdd
from functools import reduce


def get_component_types_dict(graph):
    result = {}

    try:
        vertex_dict = dict(zip(graph.vs["name"], graph.vs["compType"]))
        if 's' in vertex_dict:
            del vertex_dict['s']
        if 't' in vertex_dict:
            del vertex_dict['t']
        result.update(vertex_dict)
    except KeyError:
        pass

    try:
        edge_dict = dict(zip(graph.es["name"], graph.es["compType"]))
        result.update(edge_dict)
    except KeyError:
        pass

    return result


def egcut(g, start, terminate, vs=None, tt=None, cutsets=None):
    """
    Based on Louis Aslett's igraph based implementation of the EGCUT algorithm from Shin & Koh (1998) in the
    ReliabilityTheory R package.
    :param g: The graph from which the cutsets are to computed.
    :param start: The name of the start vertex.
    :param terminate: The name of the end vertex.
    :param vs: The vertices set in the parent node of node i of the cutset generation tree.
    :param vt: The vertices in the maximal connected component containing terminal t in the
    subgraph [v - vs, E(v - vs)].
    :param cutsets: The cutsets computed so far.
    :return: The complete set of cutsets.
    """

    if vs is None:
        vs = set([g.vs.find(name=start).index, ])
    if tt is None:
        tt = set([g.vs.find(name=terminate).index, ])
    if cutsets is None:
        cutsets = []

    v = set(range(0, g.vcount()))

    # 1.
    vx = set((v for v_neighbors in g.neighborhood(vs, order=1) for v in v_neighbors)) - vs
    # 2.
    gt = g.induced_subgraph(v - vs)
    vt = set(
        [g.vs.find(name=n).index for n in gt.vs[gt.subcomponent(gt.vs.find(name=terminate).index, mode=OUT)]["name"]])
    # 3.
    z = (v - vs) - vt
    # 4.
    if len(z & tt) != 0:
        return cutsets
    # 5.
    vs = vs | z
    # 6.
    vx = vx - z
    # 7.
    ec = [edge["name"] for edge in g.es if (edge.source in vs and edge.target not in vs)
          or (edge.target in vs and edge.source not in vs)]
    # 8.
    cutsets.append(ec)
    # 9.
    tpri = set()
    # 10.
    vx_without_t = vx - tt
    while len(vx_without_t) > 0:
        # 11.
        vertex = vx_without_t.pop()
        #  12.
        egcut(g, start, terminate, vs | set([vertex, ]), tt | tpri, cutsets)
        # 13.
        tpri = tpri | set([vertex, ])

    return cutsets


def minimal_edge_cutsets(graph, start, terminate):
    return egcut(graph, start, terminate)


def minimal_vertex_pair_cutsets(graph, start, terminate):
    """
    Gives the s-t cutsets from a graph assuming named vertices are imperfect and all edges are perfect.
    Based on Louis Aslett's algorithm given in the ReliabilityTheory R Package -
    https://github.com/louisaslett/ReliabilityTheory.
    :param graph: An igraph instance of a directed or undirected graph with a vertex named 's', a
    vertex named 't' and zero or more other named vertices
    :return: A list of tuples, where each tuple contains the names of the vertices in a minimal s-t cutset.
    """

    cutsets = graph.all_minimal_st_separators()  # All cutsets that separate the graph into connected components.

    # Remove cutsets containing s or t since they are definitely not s-t cutsets.
    s = graph.vs.find(name=start)
    t = graph.vs.find(name=terminate)
    cutsets = [cutset for cutset in cutsets if s.index not in cutset and t.index not in cutset]
    # Create list of s-t cutsets (as tuples of vertex names not indices) from cutsets by
    # appending only those that separate s and t.
    st_cutsets = []
    for i in range(0, len(cutsets)):
        g = graph.copy()
        g.delete_vertices(cutsets[i])
        s = g.vs.find(name=start)
        t = g.vs.find(name=terminate)
        if t.index not in g.subcomponent(s.index, mode="out"):
            st_cutsets.append(graph.vs(cutsets[i])["name"])

    return st_cutsets


def get_st_bdd_from_cutsets(graph):
    cutsets = minimal_vertex_pair_cutsets(graph, "s", "t")
    components = [name for name in graph.vs["name"] if name != "s" and name != "t"]

    bdd = _bdd.BDD()
    for component in components:
        bdd.add_var(component)

    cutset_bdds = []
    for cutset in cutsets:
        cutset_bdds.append(reduce(lambda x, y: bdd.apply("|", x, y), [bdd.var(component) for component in cutset]))

    root = reduce(lambda x, y: bdd.apply("&", x, y), cutset_bdds, bdd.true)

    bdd.collect_garbage()

    return bdd._bdd, root.node
