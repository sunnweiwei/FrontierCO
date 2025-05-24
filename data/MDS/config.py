import networkx as nx
import os
import pathlib
import pickle

DESCRIPTION = '''The Minimum Dominant Set (MDS) problem is a fundamental NP-hard optimization problem in graph theory. Given an undirected graph G = (V, E), where V is a set of vertices and E is a set of edges, the goal is to find the smallest subset D ⊆ V such that every vertex in V is either in D or adjacent to at least one vertex in D.'''


def solve(**kwargs):
    """
    Solve the Minimum Dominant Set problem for a given test case.

    Input:
        kwargs (dict): A dictionary with the following keys:
            - graph (networkx.Graph): The graph to solve

    Returns:
        dict: A solution dictionary containing:
            - mds_nodes (list): List of node indices in the minimum dominant set
    """
    # TODO: Implement your MDS solving algorithm here. Below is a placeholder.
    # Your function must yield multiple solutions over time, not just return one solution
    # Use Python's yield keyword repeatedly to produce a stream of solutions
    # Each yielded solution should be better than the previous one
    while True:
        yield {
        'mds_nodes': [0, 1, ...],
    }


def load_data(file_path):
    """
    Load test data for an MDS instance (same API as before).

    Args:
        file_path (str or pathlib.Path): Path to the .gr file.

    Returns:
        list[dict]: [{'graph': nx.Graph}]
    """
    file_path = pathlib.Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    G = nx.Graph()
    edges = []                     # collect edges, add in batch (fast)

    with file_path.open('r') as f:
        for line in f:
            if not line or line[0].isspace():      # skip blanks quickly
                continue

            if line[0] == 'p':                     # “p ds NODES EDGES”
                _, fmt, n_nodes, *_ = line.split()
                if fmt != 'ds':
                    raise ValueError(f"Unexpected format: {fmt}")
                G.add_nodes_from(range(1, int(n_nodes) + 1))
                continue

            # Otherwise it must be an edge line: "u v"
            u_str, v_str = line.split()
            edges.append((int(u_str), int(v_str)))

    G.add_edges_from(edges)        # one shot edge insertion

    return [{'graph': G}]

def eval_func(**kwargs):
    """
    Evaluate a Minimum Dominant Set solution for correctness.

    Args:
        graph (networkx.Graph): The graph that was solved
        mds_nodes (list): List of nodes claimed to be in the minimum dominant set

    Returns:
        int: The size of the valid dominant set, or raises an exception if invalid
    """
    graph = kwargs['graph']
    mds_nodes = kwargs['mds_nodes']

    # Check if mds_nodes is a list
    if not isinstance(mds_nodes, list):
        raise Exception("mds_nodes must be a list")

    # Check if all nodes in mds_nodes exist in the graph
    node_set = set(graph.nodes())
    for node in mds_nodes:
        if node not in node_set:
            raise Exception(f"Node {node} in solution does not exist in graph")

    # Check for duplicates in mds_nodes
    if len(mds_nodes) != len(set(mds_nodes)):
        raise Exception("Duplicate nodes in solution")

    # Get the actual size
    actual_size = len(mds_nodes)

    # Most important: Check if it's a dominant set (every node is either in the set or adjacent to a node in the set)
    dominated_nodes = set(mds_nodes)  # Nodes in the set
    
    # Add all neighbors of nodes in the set
    for node in mds_nodes:
        dominated_nodes.update(graph.neighbors(node))
    
    # Check if all nodes are dominated
    if dominated_nodes != node_set:
        undominated = node_set - dominated_nodes
        raise Exception(f"Not a dominant set: nodes {undominated} are not dominated")

    return actual_size


def norm_score(results):
    optimal_scores = {'easy_test_instances/exact_066.gr': [707.0], 'easy_test_instances/exact_088.gr': [707.0], 'easy_test_instances/exact_075.gr': [706.0], 'easy_test_instances/exact_093.gr': [706.0], 'easy_test_instances/exact_097.gr': [706.0], 'easy_test_instances/exact_081.gr': [1216.0], 'easy_test_instances/exact_057.gr': [705.0], 'easy_test_instances/exact_063.gr': [805.0], 'easy_test_instances/exact_072.gr': [805.0], 'easy_test_instances/exact_092.gr': [1183.0], 'easy_test_instances/exact_069.gr': [1171.0], 'easy_test_instances/exact_033.gr': [5539.0], 'easy_test_instances/exact_071.gr': [2689.0], 'easy_test_instances/exact_051.gr': [849.0], 'easy_test_instances/exact_067.gr': [989.0], 'easy_test_instances/exact_076.gr': [1597.0], 'easy_test_instances/exact_058.gr': [740.0], 'easy_test_instances/exact_056.gr': [1512.0], 'easy_test_instances/exact_083.gr': [1866.0], 'easy_test_instances/exact_034.gr': [5842.0], 'hard_test_instances/heuristic_049.gr': [3062.0], 'hard_test_instances/heuristic_065.gr': [3159.0], 'hard_test_instances/heuristic_016.gr': [3352.0], 'hard_test_instances/heuristic_042.gr': [2999.0], 'hard_test_instances/heuristic_017.gr': [3330.0], 'hard_test_instances/heuristic_019.gr': [3062.0], 'hard_test_instances/heuristic_036.gr': [3050.0], 'hard_test_instances/heuristic_067.gr': [3277.0], 'hard_test_instances/heuristic_097.gr': [3025.0], 'hard_test_instances/heuristic_015.gr': [3077.0], 'hard_test_instances/heuristic_059.gr': [2997.0], 'hard_test_instances/heuristic_037.gr': [3054.0], 'hard_test_instances/heuristic_026.gr': [3025.0], 'hard_test_instances/heuristic_060.gr': [3001.0], 'hard_test_instances/heuristic_078.gr': [2829.0], 'hard_test_instances/heuristic_044.gr': [2937.0], 'hard_test_instances/heuristic_003.gr': [637607.0], 'hard_test_instances/heuristic_066.gr': [1047.0], 'hard_test_instances/heuristic_074.gr': [331531.0], 'hard_test_instances/heuristic_077.gr': [427644.0], 'valid_instances/ba_graph_large_train_12.txt': [96.0], 'valid_instances/ba_graph_large_train_11.txt': [93.0], 'valid_instances/ba_graph_large_train_10.txt': [123.0], 'valid_instances/ba_graph_large_train_19.txt': [116.0], 'valid_instances/ba_graph_large_train_14.txt': [93.0], 'valid_instances/ba_graph_large_train_0.txt': [118.0], 'valid_instances/ba_graph_large_train_17.txt': [106.0], 'valid_instances/ba_graph_large_train_6.txt': [107.0], 'valid_instances/ba_graph_large_train_18.txt': [117.0], 'valid_instances/ba_graph_large_train_13.txt': [120.0], 'valid_instances/ba_graph_large_train_7.txt': [86.0], 'valid_instances/ba_graph_large_train_5.txt': [114.0], 'valid_instances/ba_graph_large_train_3.txt': [118.0], 'valid_instances/ba_graph_large_train_9.txt': [114.0], 'valid_instances/ba_graph_large_train_15.txt': [92.0], 'valid_instances/ba_graph_large_train_16.txt': [112.0], 'valid_instances/ba_graph_large_train_8.txt': [124.0], 'valid_instances/ba_graph_large_train_2.txt': [116.0], 'valid_instances/ba_graph_large_train_4.txt': [121.0], 'valid_instances/ba_graph_large_train_1.txt': [124.0]}
    # print(results)
    normed = {}
    for case, (scores, error_message) in results.items():
        if case not in optimal_scores:
            continue  # Skip if there's no optimal score defined.
        optimal_list = optimal_scores[case]
        normed_scores = []
        # Compute normalized score for each index.
        for idx, score in enumerate(scores):
            if isinstance(score, (int, float)):
                normed_scores.append(1 - abs(score - optimal_list[idx]) / max(score, optimal_list[idx]))
            else:
                normed_scores.append(score)
        normed[case] = (normed_scores, error_message)

    return normed
