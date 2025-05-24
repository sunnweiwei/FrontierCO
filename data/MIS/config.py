import networkx as nx
import os
import pathlib
import pickle

DESCRIPTION = '''The Maximum Independent Set (MIS) problem is a fundamental NP-hard optimization problem in graph theory. Given an undirected graph G = (V, E), where V is a set of vertices and E is a set of edges, the goal is to find the largest subset S ⊆ V such that no two vertices in S are adjacent (i.e., connected by an edge).'''


def solve(**kwargs):
    """
    Solve the Maximum Independent Set problem for a given test case.

   Input:
        kwargs (dict): A dictionary with the following keys:
            - graph (networkx.Graph): The graph to solve

    Returns:
        dict: A solution dictionary containing:
            - mis_nodes (list): List of node indices in the maximum independent set
    """
    # TODO: Implement your MIS solving algorithm here. Below is a placeholder.
    # Your function must yield multiple solutions over time, not just return one solution
    # Use Python's yield keyword repeatedly to produce a stream of solutions
    # Each yielded solution should be better than the previous one
    while True:
        yield {
            'mis_nodes': [0, 1, ...],
        }


def load_data(file_path):
    """
    Load test data for MIS problem
    
    Args:
        file_path (str or pathlib.Path): Path to the file
        
    Returns:
        dict: A dictionary containing a test case with graph data
    """
    file_path = pathlib.Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.suffix != '.mis':
        raise ValueError(f"Expected .dimacs file, got {file_path.suffix}")

    try:
        # Create an empty graph
        G = nx.Graph()

        with open(file_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Parse the line
            parts = line.split()

            # Problem line (p edge NODES EDGES)
            if parts[0] == 'p' and parts[1] == 'edge':
                num_nodes = int(parts[2])
                # Pre-add all nodes
                G.add_nodes_from(range(1, num_nodes + 1))

            # Edge line (e NODE1 NODE2)
            elif parts[0] == 'e':
                node1 = int(parts[1])
                node2 = int(parts[2])
                G.add_edge(node1, node2)

        # Create a test case dictionary
        test_case = {
            'graph': G
        }

        return [test_case]

    except Exception as e:
        raise Exception(f"Error loading graph from {file_path}: {e}")


def eval_func(**kwargs):
    """
    Evaluate a Maximum Independent Set solution for correctness.

    Args:
        name (str): Name of the test case
        graph (networkx.Graph): The graph that was solved
        mis_nodes (list): List of nodes claimed to be in the maximum independent set
        mis_size (int): Claimed size of the maximum independent set

    Returns:
        dict: Evaluation results containing:
            - is_valid (bool): Whether the solution is a valid independent set
            - actual_size (int): The actual size of the provided solution
            - score (int): The score of the solution (0 if invalid, actual_size if valid)
            - error (str, optional): Error message if any constraint is violated
    """

    graph = kwargs['graph']
    mis_nodes = kwargs['mis_nodes']

    # Check if mis_nodes is a list
    if not isinstance(mis_nodes, list):
        raise Exception("mis_nodes must be a list")

    # Check if all nodes in mis_nodes exist in the graph
    node_set = set(graph.nodes())
    for node in mis_nodes:
        if node not in node_set:
            raise Exception(f"Node {node} in solution does not exist in graph")

    # Check for duplicates in mis_nodes
    if len(mis_nodes) != len(set(mis_nodes)):
        raise Exception("Duplicate nodes in solution")

    # Check if mis_size matches the length of mis_nodes
    actual_size = len(mis_nodes)

    # Most important: Check if it's an independent set (no edges between any nodes)
    for i in range(len(mis_nodes)):
        for j in range(i + 1, len(mis_nodes)):
            if graph.has_edge(mis_nodes[i], mis_nodes[j]):
                raise Exception(f"Not an independent set: edge exists between {mis_nodes[i]} and {mis_nodes[j]}")

    return actual_size

def norm_score(results):
    optimal_scores = {'easy_test_instances/C1000.9.mis': [68.0], 'easy_test_instances/C125.9.mis': [34.0], 'easy_test_instances/C2000.5.mis': [16.0], 'easy_test_instances/C2000.9.mis': [80.0], 'easy_test_instances/C250.9.mis': [44.0], 'easy_test_instances/C4000.5.mis': [18.0], 'easy_test_instances/C500.9.mis': [57.0], 'easy_test_instances/DSJC1000.5.mis': [15.0], 'easy_test_instances/DSJC500.5.mis': [13.0], 'easy_test_instances/MANN_a27.mis': [126.0], 'easy_test_instances/MANN_a45.mis': [345.0], 'easy_test_instances/MANN_a81.mis': [1100.0], 'easy_test_instances/brock200_2.mis': [12.0], 'easy_test_instances/brock200_4.mis': [17.0], 'easy_test_instances/brock400_2.mis': [29.0], 'easy_test_instances/brock400_4.mis': [33.0], 'easy_test_instances/brock800_2.mis': [24.0], 'easy_test_instances/brock800_4.mis': [26.0], 'easy_test_instances/gen200_p0.9_44.mis': [44.0], 'easy_test_instances/gen200_p0.9_55.mis': [55.0], 'easy_test_instances/gen400_p0.9_55.mis': [55.0], 'easy_test_instances/gen400_p0.9_65.mis': [65.0], 'easy_test_instances/gen400_p0.9_75.mis': [75.0], 'easy_test_instances/hamming10-4.mis': [40.0], 'easy_test_instances/hamming8-4.mis': [16.0], 'easy_test_instances/keller4.mis': [11.0], 'easy_test_instances/keller5.mis': [27.0], 'easy_test_instances/keller6.mis': [59.0], 'easy_test_instances/p_hat1500-1.mis': [12.0], 'easy_test_instances/p_hat1500-2.mis': [65.0], 'easy_test_instances/p_hat1500-3.mis': [94.0], 'easy_test_instances/p_hat300-1.mis': [8.0], 'easy_test_instances/p_hat300-2.mis': [25.0], 'easy_test_instances/p_hat300-3.mis': [36.0], 'easy_test_instances/p_hat700-1.mis': [11.0], 'easy_test_instances/p_hat700-2.mis': [44.0], 'easy_test_instances/p_hat700-3.mis': [62.0]}
    optimal_scores = optimal_scores | {'hard_test_instances/frb100-40.mis': [98.0], 'hard_test_instances/frb50-23-1.mis': [50.0], 'hard_test_instances/frb50-23-2.mis': [50.0], 'hard_test_instances/frb50-23-3.mis': [50.0], 'hard_test_instances/frb50-23-4.mis': [50.0], 'hard_test_instances/frb50-23-5.mis': [50.0], 'hard_test_instances/frb53-24-1.mis': [53.0], 'hard_test_instances/frb53-24-2.mis': [53.0], 'hard_test_instances/frb53-24-3.mis': [53.0], 'hard_test_instances/frb53-24-4.mis': [53.0], 'hard_test_instances/frb53-24-5.mis': [53.0], 'hard_test_instances/frb59-26-1.mis': [59.0], 'hard_test_instances/frb59-26-2.mis': [59.0], 'hard_test_instances/frb59-26-3.mis': [59.0], 'hard_test_instances/frb59-26-4.mis': [59.0], 'hard_test_instances/frb59-26-5.mis': [59.0]}
    optimal_scores = optimal_scores | {'valid_instances/RB_800_1200_0.mis': [47.0], 'valid_instances/RB_800_1200_1.mis': [50.0], 'valid_instances/RB_800_1200_10.mis': [37.0], 'valid_instances/RB_800_1200_11.mis': [50.0], 'valid_instances/RB_800_1200_12.mis': [49.0], 'valid_instances/RB_800_1200_13.mis': [44.0], 'valid_instances/RB_800_1200_14.mis': [41.0], 'valid_instances/RB_800_1200_15.mis': [45.0], 'valid_instances/RB_800_1200_16.mis': [43.0], 'valid_instances/RB_800_1200_17.mis': [40.0], 'valid_instances/RB_800_1200_18.mis': [40.0], 'valid_instances/RB_800_1200_19.mis': [36.0], 'valid_instances/RB_800_1200_2.mis': [36.0], 'valid_instances/RB_800_1200_3.mis': [50.0], 'valid_instances/RB_800_1200_4.mis': [44.0], 'valid_instances/RB_800_1200_5.mis': [47.0], 'valid_instances/RB_800_1200_6.mis': [45.0], 'valid_instances/RB_800_1200_7.mis': [38.0], 'valid_instances/RB_800_1200_8.mis': [38.0], 'valid_instances/RB_800_1200_9.mis': [50.0]}

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
