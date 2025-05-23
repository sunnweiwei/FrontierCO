import numpy as np
import networkx as nx
import argparse
import os
import math
import random

# Unified random graph generator
def generate_random_graph(rng, n_nodes, n_edges, special_vertices=None):
    """
    Generate a random graph with the given parameters.

    Args:
        rng: Random number generator
        n_nodes: Number of nodes
        n_edges: Number of edges
        special_vertices: List of special vertices (optional)

    Returns:
        networkx.Graph: A random connected graph
    """
    # Create nodes
    G = nx.Graph()
    G.add_nodes_from(range(1, n_nodes + 1))  # Node indices start from 1

    # Generate random spanning tree
    nodes = list(G.nodes())

    # Start with node 1 included
    included = [nodes[0]]
    remaining = nodes[1:]

    # Generate spanning tree using Prim's algorithm with random weights
    while remaining:
        u = rng.choice(included)
        v = rng.choice(remaining)
        G.add_edge(u, v)
        included.append(v)
        remaining.remove(v)
        
    # Add additional random edges until reaching n_edges
    possible_edges = [(u, v) for u in G.nodes() for v in G.nodes() if u < v and not G.has_edge(u, v)]
    additional_edges_count = n_edges - (n_nodes - 1)

    if additional_edges_count > 0:
        # Randomly select additional edges
        if len(possible_edges) < additional_edges_count:
            print(f"Warning: Can only add {len(possible_edges)} more edges instead of requested {additional_edges_count}")
            additional_edges_count = len(possible_edges)

            indices = rng.choice(len(all_possible_edges), size=additional_edges, replace=False)
            edges_to_add = [all_possible_edges[i] for i in indices]
            G.add_edges_from(edges_to_add)

    return G

# Assign weights to edges based on problem type
def assign_weights(G, problem_type, rng, special_vertices=None, **kwargs):
    """
    Assign weights to edges based on the problem type.
    
    Args:
        G: networkx.Graph - The graph to assign weights to
        problem_type: str - 'random', 'euclidean', or 'incidence'
        rng: Random number generator
        special_vertices: List of special vertices (for incidence problems)
        **kwargs: Additional parameters for specific problem types
        
    Returns:
        networkx.Graph: The graph with weights assigned
    """
    if problem_type == 'random':
        # Random weights [1, 100]
        for u, v in G.edges():
            G[u][v]['weight'] = rng.randint(1, 101)
    
    elif problem_type == 'euclidean':
        # Generate random coordinates for each vertex
        coords = {}
        for v in G.nodes():
            coords[v] = (rng.uniform(0, 500), rng.uniform(0, 500))
            G.nodes[v]['pos'] = coords[v]
        
        # Assign Euclidean distances as weights
        for u, v in G.edges():
            x1, y1 = coords[u]
            x2, y2 = coords[v]
            dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            G[u][v]['weight'] = math.ceil(dist)  # Round up to integer
    
    elif problem_type == 'incidence':
        # Default standard deviation
        std_dev = kwargs.get('std_dev', 5)
        
        if special_vertices is None or len(special_vertices) == 0:
            # If no special vertices provided, use a default (e.g., first 10% of vertices)
            n_special = max(1, int(len(G.nodes) * 0.1))
            special_vertices = list(range(1, n_special + 1))
        
        # Assign weights based on incidence with special vertices
        for u, v in G.edges():
            # Count how many endpoints are special vertices
            special_count = sum(1 for node in [u, v] if node in special_vertices)
            
            # Determine mean based on incidence
            if special_count == 0:
                mean = 100  # No incidence with special vertices
            elif special_count == 1:
                mean = 200  # One endpoint is a special vertex
            else:
                mean = 300  # Both endpoints are special vertices
            
            # Sample from normal distribution
            r = rng.normal(mean, std_dev)
            # Apply constraints
            G[u][v]['weight'] = min(max(1, round(r)), 500)
    
    elif problem_type == 'rectilinear':
        # For rectilinear graphs, a different graph generation approach is needed
        # This method just adds placeholder weights if called on a non-rectilinear graph
        for u, v in G.edges():
            G[u][v]['weight'] = rng.randint(1, 501)
    
    return G

# Generate graph instances with specific parameters
def generate_graph_instance(rng, problem_type, n_vertices, n_edges, n_terminals, **kwargs):
    """
    Generate a graph instance based on the problem type.
    
    Args:
        rng: Random number generator
        problem_type: Problem type ('random', 'euclidean', 'incidence')
        n_vertices: Number of vertices
        n_edges: Number of edges
        n_terminals: Number of terminals (special vertices)
        **kwargs: Additional parameters
        
    Returns:
        tuple: (G, terminals) where G is a networkx Graph and terminals is a list
    """
    # Generate terminals (special vertices)
    if callable(n_terminals):
        # If n_terminals is a function, calculate based on n_vertices
        num_terminals = n_terminals(n_vertices)
    else:
        num_terminals = n_terminals
    
    num_terminals = min(num_terminals, n_vertices)
    terminals = sorted(rng.choice(range(1, n_vertices + 1), size=num_terminals, replace=False))
    
    # Generate the basic graph structure
    G = generate_random_graph(rng, n_vertices, n_edges, terminals)
    
    # Assign weights based on problem type
    G = assign_weights(G, problem_type, rng, terminals, **kwargs)
    
    return G, terminals

# Generate a hypercube graph instance
def generate_hypercube_instance(rng, d):
    """
    Generate a Hypercube graph with the given parameters.
    
    Args:
        rng: Random number generator
        d: Dimension of the hypercube
        
    Returns:
        tuple: (G, terminals) where G is a networkx Graph and terminals is a list
    """
    G = nx.Graph()
    
    # Add vertices
    for i in range(2**d):
        G.add_node(i + 1)  # Use 1-based indexing
        
        # Connect to neighbors
        for k in range(d):
            j = i ^ (1 << k)  # XOR
            if i < j:
                G.add_edge(i + 1, j + 1, weight=rng.randint(100, 111))
    
    # Terminals = nodes with even parity
    terminals = [i + 1 for i in range(2**d) if bin(i).count('1') % 2 == 0]
    
    return G, terminals

# Generate a rectilinear graph instance
def generate_rectilinear_instance(rng, num_lines, n_terminals):
    """
    Generate a rectilinear graph instance.
    
    Args:
        rng: Random number generator
        num_lines: Number of horizontal/vertical lines
        n_terminals: Number of terminals or function to calculate it
        
    Returns:
        tuple: (G, terminals) where G is a networkx Graph and terminals is a list
    """
    G = nx.Graph()
    
    # Generate random distances between lines [1, 500]
    h_distances = [rng.randint(1, 501) for _ in range(num_lines-1)]
    v_distances = [rng.randint(1, 501) for _ in range(num_lines-1)]
    
    # Calculate coordinates of grid points
    h_coords = [0] + [sum(h_distances[:i]) for i in range(1, num_lines)]
    v_coords = [0] + [sum(v_distances[:i]) for i in range(1, num_lines)]
    
    # Add nodes with coordinates
    for i in range(num_lines):
        for j in range(num_lines):
            node_id = i * num_lines + j + 1
            G.add_node(node_id, pos=(h_coords[i], v_coords[j]))
    
    # Add horizontal edges
    for i in range(num_lines):
        for j in range(num_lines-1):
            v1_id = i * num_lines + j + 1
            v2_id = i * num_lines + (j + 1) + 1
            G.add_edge(v1_id, v2_id, weight=h_distances[j])
    
    # Add vertical edges
    for i in range(num_lines-1):
        for j in range(num_lines):
            v1_id = i * num_lines + j + 1
            v2_id = (i + 1) * num_lines + j + 1
            G.add_edge(v1_id, v2_id, weight=v_distances[i])
    
    # Calculate terminals
    if callable(n_terminals):
        num_terminals = n_terminals(num_lines * num_lines)
    else:
        num_terminals = n_terminals
    
    num_terminals = min(num_terminals, num_lines * num_lines)
    
    # Special rule for selecting terminals in rectilinear graphs
    # Try to select them on new horizontal and vertical lines when possible
    used_h_lines = set()
    used_v_lines = set()
    terminals = []
    
    # First, select vertices on new lines when possible
    while len(terminals) < num_terminals and len(used_h_lines) < num_lines and len(used_v_lines) < num_lines:
        # Find unused lines
        available_h = [h for h in range(num_lines) if h not in used_h_lines]
        available_v = [v for v in range(num_lines) if v not in used_v_lines]
        
        if not available_h or not available_v:
            break
            
        h = rng.choice(available_h)
        v = rng.choice(available_v)
        
        node = h * num_lines + v + 1
        if node not in terminals:
            terminals.append(node)
            used_h_lines.add(h)
            used_v_lines.add(v)
    
    # Fill remaining terminals randomly if needed
    if len(terminals) < num_terminals:
        remaining = [n for n in range(1, num_lines*num_lines + 1) if n not in terminals]
        additional = sorted(rng.choice(remaining, size=min(num_terminals - len(terminals), len(remaining)), replace=False))
        terminals.extend(additional)
    
    terminals.sort()
    
    return G, terminals

# Function to save graph in STP format
def save_stp(filename, name, creator, remark, nodes, edges, terminals):
    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
    
    with open(filename, 'w') as f:
        f.write("33D32945 STP File, STP Format Version 1.0\n\n")
        f.write("SECTION Comment\n")
        f.write(f'Name    "{name}"\n')
        f.write(f'Creator "{creator}"\n')
        f.write(f'Remark  "{remark}"\n')
        f.write("END\n\n")
        
        f.write("SECTION Graph\n")
        f.write(f"Nodes {nodes}\n")
        f.write(f"Edges {len(edges)}\n")
        
        for u, v, weight in edges:
            f.write(f"E {u} {v} {weight}\n")
        f.write("END\n\n")
        
        f.write("SECTION Terminals\n")
        f.write(f"Terminals {len(terminals)}\n")
        
        for t in sorted(terminals):
            f.write(f"T {t}\n")
        f.write("END\n")
        
        f.write("EOF\n")

# Calculate edge density based on formula
def calc_num_edges(n_nodes, density):
    if density == '3v/2':
        return int(3 * n_nodes / 2)
    elif density == '2v':
        return 2 * n_nodes
    elif density == 'v*log(v)':
        return int(n_nodes * np.log(n_nodes))
    elif density == '2v*log(v)':
        return int(2 * n_nodes * np.log(n_nodes))
    elif density == 'v*(v-1)/4':
        return int(n_nodes * (n_nodes - 1) / 4)
    return 2 * n_nodes  # Default

def main():
    parser = argparse.ArgumentParser(description='Generate Steiner Tree Problem instances')
    
    parser.add_argument('--type', choices=['random', 'euclidean', 'hypercube', 'incidence', 'rectilinear', 'all'], 
                        default='all', help='Type of problem to generate (default: all)')
    parser.add_argument('--nodes', type=int, nargs='+', default=[80, 160, 320],
                        help='Node counts (default: [80, 160, 320])')
    parser.add_argument('--dimensions', type=int, nargs='+', default=[6, 7, 8, 9, 10],
                        help='Dimensions for hypercube instances (default: 6-10)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--instances_per_config', type=int, default=1,
                        help='Number of instances to generate per configuration (default: 1)')
    parser.add_argument('--output_dir', default='stp_instances',
                        help='Output directory (default: stp_instances)')

    args = parser.parse_args()
    
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    types = [args.type] if args.type != 'all' else ['random', 'euclidean', 'hypercube', 'incidence', 'rectilinear']
    count = 0
    
    # Terminal functions
    k_funcs = {
        'log(v)': lambda v: max(1, int(np.log2(v))),
        'sqrt(v)': lambda v: max(1, int(np.sqrt(v))),
        '2sqrt(v)': lambda v: max(1, int(2*np.sqrt(v))),
        'v/4': lambda v: max(1, v//4)
    }
    
    # Edge densities
    densities = ['v*log(v)', '2v*log(v)']#['3v/2', '2v']#, 'v*log(v)', '2v*log(v)']#, 'v*(v-1)/4']
    
    # Random instances
    if 'random' in types:
        dir_path = args.output_dir
        os.makedirs(dir_path, exist_ok=True)
        
        for n_nodes in args.nodes:
            for k_name, k_func in k_funcs.items():
                n_terminals = k_func(n_nodes)
                
                for density in densities:
                    # Calculate number of edges
                    n_edges = calc_num_edges(n_nodes, density)
                    
                    # Generate 5 variants
                    for variant in range(args.instances_per_config):
                        rng = np.random.RandomState(args.seed)
                        
                        G, terminals = generate_graph_instance(
                            rng=rng,
                            problem_type='random',
                            n_vertices=n_nodes,
                            n_edges=n_edges,
                            n_terminals=n_terminals
                        )
                        
                        edge_list = [(u, v, G[u][v]['weight']) for u, v in G.edges()]
                        
                        k_name_clean = k_name.replace('(', '').replace(')', '').replace('*', '').replace('/', '-')
                        density_clean = density.replace('*', '').replace('/', '-').replace('(', '').replace(')', '')
                        name = f"r{n_nodes}_{k_name_clean}_{density_clean}_{variant}"
                        filename = f"{dir_path}/{name}.stp"
                        
                        save_stp(
                            filename, name, "Steiner Tree Generator (Random)", 
                            f"Random: |V|={n_nodes}, |T|={len(terminals)}, |E|={len(edge_list)}, k={k_name}, density={density}",
                            n_nodes, edge_list, terminals
                        )
                        
                        print(f"Generated: {name}")
                        count += 1
    
    # Euclidean instances
    if 'euclidean' in types:
        dir_path = args.output_dir
        os.makedirs(dir_path, exist_ok=True)
        
        for n_nodes in args.nodes:
            for k_name, k_func in k_funcs.items():
                n_terminals = k_func(n_nodes)
                
                for density in densities:
                    # Calculate number of edges
                    n_edges = calc_num_edges(n_nodes, density)
                    
                    # Generate 5 variants
                    for variant in range(args.instances_per_config):
                        rng = np.random.RandomState(args.seed)
                        
                        G, terminals = generate_graph_instance(
                            rng=rng,
                            problem_type='euclidean',
                            n_vertices=n_nodes,
                            n_edges=n_edges,
                            n_terminals=n_terminals
                        )
                        
                        edge_list = [(u, v, G[u][v]['weight']) for u, v in G.edges()]
                        
                        k_name_clean = k_name.replace('(', '').replace(')', '').replace('*', '').replace('/', '-')
                        density_clean = density.replace('*', '').replace('/', '-').replace('(', '').replace(')', '')
                        name = f"e{n_nodes}_{k_name_clean}_{density_clean}_{variant}"
                        filename = f"{dir_path}/{name}.stp"
                        
                        save_stp(
                            filename, name, "Steiner Tree Generator (Euclidean)", 
                            f"Euclidean: |V|={n_nodes}, |T|={len(terminals)}, |E|={len(edge_list)}, k={k_name}, density={density}",
                            n_nodes, edge_list, terminals
                        )
                        
                        print(f"Generated: {name}")
                        count += 1
    
    # Hypercube instances
    if 'hypercube' in types:
        dir_path = args.output_dir
        os.makedirs(dir_path, exist_ok=True)
        
        for d in args.dimensions:
            for variant in range(args.instances_per_config):
                rng = np.random.RandomState(args.seed)
                
                G, terminals = generate_hypercube_instance(rng, d)
                edge_list = [(u, v, G[u][v]['weight']) for u, v in G.edges()]
                
                name = f"hc{d}_{variant}p"
                filename = f"{dir_path}/{name}.stp"
                
                save_stp(
                    filename, name, "Steiner Tree Generator (Hypercube)", 
                    f"Hypercube: d={d}, |V|={len(G.nodes())}, |T|={len(terminals)}, |E|={len(G.edges())}",
                    len(G.nodes()), edge_list, terminals
                )
                
                print(f"Generated: {name}")
                count += 1
    
    # Incidence instances
    if 'incidence' in types:
        dir_path = args.output_dir
        os.makedirs(dir_path, exist_ok=True)
                
        for n_nodes in args.nodes:
            for k_name, k_func in k_funcs.items():
                n_terminals = k_func(n_nodes)
                
                for density in densities:
                    # Calculate number of edges
                    n_edges = calc_num_edges(n_nodes, density)
                    
                    # Generate 5 variants for each configuration
                    for variant in range(args.instances_per_config):
                        rng = np.random.RandomState(args.seed)
                        
                        G, terminals = generate_graph_instance(
                            rng=rng,
                            problem_type='incidence',
                            n_vertices=n_nodes,
                            n_edges=n_edges,
                            n_terminals=n_terminals,
                            std_dev=5
                        )
                        
                        edge_list = [(u, v, G[u][v]['weight']) for u, v in G.edges()]
                        
                        k_name_clean = k_name.replace('(', '').replace(')', '').replace('*', '').replace('/', '-')
                        density_clean = density.replace('*', '').replace('/', '-').replace('(', '').replace(')', '')
                        name = f"i{n_nodes}_{k_name_clean}_{density_clean}_{variant}p"
                        filename = f"{dir_path}/{name}.stp"
                        
                        save_stp(
                            filename, name, "Steiner Tree Generator (Incidence)", 
                            f"Incidence: |V|={n_nodes}, |T|={len(terminals)}, |E|={len(edge_list)}, k={k_name}, density={density}",
                            n_nodes, edge_list, terminals
                        )
                        
                        print(f"Generated: {name}")
                        count += 1
    
    # Rectilinear instances
    if 'rectilinear' in types:
        dir_path = args.output_dir
        os.makedirs(dir_path, exist_ok=True)
        
        # Line counts for rectilinear graphs
        line_counts = [9, 13, 18]
        
        for l in line_counts:
            # Special vertex densities
            special_densities = [max(1, l-3), l, l+3, 2*l, 3*l]
            
            for k in special_densities:
                # Generate 5 variants
                for variant in range(args.instances_per_config):
                    rng = np.random.RandomState(args.seed + 4000 + l + k + variant)
                    
                    G, terminals = generate_rectilinear_instance(rng, l, k)
                    edge_list = [(u, v, G[u][v]['weight']) for u, v in G.edges()]
                    
                    name = f"rl{l}_{k}_{variant}"
                    filename = f"{dir_path}/{name}.stp"
                    
                    save_stp(
                        filename, name, "Steiner Tree Generator (Rectilinear)", 
                        f"Rectilinear: lines={l}, |V|={len(G.nodes())}, |T|={len(terminals)}, |E|={len(G.edges())}, k={k}",
                        len(G.nodes()), edge_list, terminals
                    )
                    
                    print(f"Generated: {name}")
                    count += 1
    
    print(f"\nTotal instances: {count}")
    print(f"Saved in: {args.output_dir}")

if __name__ == "__main__":
    main()