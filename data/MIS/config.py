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
    solution = {
        'mis_nodes': [0, 1, ...],
    }
    return solution


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
        
        return test_case
        
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