import networkx as nx
import numpy as np
import argparse
import os
from tqdm import tqdm

def generate_ba_graph(n, m):
    """Generate a Barabasi-Albert graph with n nodes and m edges per new node."""
    graph = nx.barabasi_albert_graph(n, m)
    return graph

def save_graph_to_file(graph, filename):
    """Save graph to file in specified format."""
    with open(filename, 'w') as f:
        # Write header
        f.write(f"p ds {graph.number_of_nodes()} {graph.number_of_edges()}\n")
        
        # Write edges (1-indexed)
        for u, v in graph.edges():
            f.write(f"{u+1} {v+1}\n")

def main():
    parser = argparse.ArgumentParser(description='Generate Barabasi-Albert graph instances')
    parser.add_argument('--dataset_name', type=str, default='large',
                        help='Dataset name (should contain "small", "large", "huge", "giant", or "dummy")')
    parser.add_argument('--instances', type=int, default=10,
                        help='Number of instances to generate (default: 10)')
    parser.add_argument('--m', type=int, default=4,
                        help='Number of edges for each new node (default: 4)')
    parser.add_argument('--output', type=str, default='training_instances',
                        help='Output directory (default: training_instances)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (default: None)')
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"Using random seed: {args.seed}")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    print(f"Generating Barabasi-Albert dataset '{args.dataset_name}' "
          f"with '{args.instances}' instances!")
    
    # Generate instances
    for idx in tqdm(range(args.instances)):
        # Determine number of nodes based on dataset name
        if "small" in args.dataset_name:
            curr_n = np.random.randint(101) + 200  # 200-300 nodes
        elif "large" in args.dataset_name:
            curr_n = np.random.randint(401) + 800  # 800-1200 nodes
        elif "huge" in args.dataset_name:
            curr_n = np.random.randint(601) + 1200  # 1200-1800 nodes
        elif "giant" in args.dataset_name:
            curr_n = np.random.randint(1001) + 2000  # 2000-3000 nodes
        elif "dummy" in args.dataset_name:
            curr_n = np.random.randint(40) + 80  # 80-120 nodes
        else:
            raise NotImplementedError('Dataset name must contain either "small", "large", "huge", "giant" or "dummy" to infer the number of nodes')
        
        # Generate BA graph
        graph = generate_ba_graph(n=curr_n, m=args.m)
        
        # Save graph to file
        filename = os.path.join(args.output, f"ba_graph_{args.dataset_name}_{idx}.gr")
        save_graph_to_file(graph, filename)
    
    print(f"All BA graphs generated in {args.output} directory")

if __name__ == "__main__":
    main()