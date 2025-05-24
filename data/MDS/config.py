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
    solution = {
        'mds_nodes': [0, 1, ...],
    }
    return solution


def load_data(file_path):
    """
    Load test data for MDS problem
    
    Args:
        file_path (str or pathlib.Path): Path to the file
        
    Returns:
        dict: A dictionary containing a test case with graph data
    """
    file_path = pathlib.Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if file_path.suffix != '.gr':
        raise ValueError(f"Expected .gr file, got {file_path.suffix}")
    
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
            
            # Problem line (p ds NODES EDGES)
            if parts[0] == 'p' and parts[1] == 'ds':
                num_nodes = int(parts[2])
                # Pre-add all nodes
                G.add_nodes_from(range(1, num_nodes + 1))
            
            # Edge line (NODE1 NODE2) - note: no 'e' prefix in this format
            elif len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                node1 = int(parts[0])
                node2 = int(parts[1])
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