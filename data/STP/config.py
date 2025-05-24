DESCRIPTION = '''The Steiner Tree Problem in Graphs requires finding a minimum-cost subgraph that spans a specified set of terminal vertices. Formally, given an undirected graph defined by its vertex count (n), edge count (m), a mapping of edge pairs to weights (graph_edges), and a list of terminal vertices (terminals), the objective is to select a subset of edges that connects all the terminal vertices while minimizing the sum of the edge weights. The solution must consist solely of edges present in the input graph and form a tree that satisfies connectivity among the terminals, with the total cost of the chosen edges equaling the declared cost. The evaluation metric is the aggregated weight of the selected edges, and the output should provide the declared total cost, the number of edges used, and the list of chosen edges.'''


def solve(**kwargs):
    """
    Solves the Steiner Tree Problem in Graphs.

    Given an undirected graph with weighted edges (provided in kwargs as:
      - n: number of vertices,
      - m: number of edges,
      - graph_edges: a dictionary mapping (min(u,v), max(u,v)) -> cost.
      - terminals: list of terminal vertices),
    the goal is to compute a subgraph (a tree) that connects all the terminal vertices
    with minimum total cost.

    Important: All vertex indices u, v, and all terminals are 1-indexed, i.e., values range from 1 to n.

    Evaluation Metric:
      The solution is scored by summing the weights of the selected edges.
      The returned solution must be valid in that:
        - All selected edges exist in the input graph.
        - The total cost (sum of the weights) equals the declared cost (within tolerance).
        - All terminal vertices are connected in the solution subgraph.

    Input kwargs: the dictionary containing keys 'n', 'm', 'graph_edges', and 'terminals'.

    Returns:
      A dictionary with the following keys:
        - 'declared_cost': Total cost of the solution (a number).
        - 'num_edges': Number of edges in the solution (an integer).
        - 'edges': A list of tuples, each tuple (u, v) representing an edge in the solution.

    NOTE: This is a placeholder implementation.
    """
    # Placeholder implementation.
    # Here you would implement your algorithm.
    # Your function must yield multiple solutions over time, not just return one solution
    # Use Python's yield keyword repeatedly to produce a stream of solutions
    # Each yielded solution should be better than the previous one
    while True:
        yield {
            'declared_cost': 0.0,  # Replace with the computed total cost.
            'num_edges': 0,  # Replace with the number of edges in your solution.
            'edges': []  # Replace with the list of edges (each as a tuple (u, v)).
        }


def load_data(file_path):
    """
    Load data from an STP file.

    Args:
        file_path: Path to the STP file

    Returns:
        Dictionary with the following keys:
        - 'n': number of vertices,
        - 'm': number of edges,
        - 'graph_edges': dictionary mapping (min(u,v), max(u,v)) -> cost,
        - 'terminals': list of terminal vertices.
    """
    # Initialize variables
    # Initialize variables
    data = {
        'n': 0,
        'm': 0,
        'graph_edges': {},
        'terminals': []
    }

    # Current section being read
    current_section = None

    with open(file_path, 'r') as f:
        for line in f:
            # Remove comments and whitespace
            if '#' in line:
                line = line[:line.index('#')]
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Check for section starts
            if line.upper().startswith('SECTION'):
                current_section = line.split()[1].upper()
                continue

            # Check for section ends
            if line.upper() == 'END':
                if current_section == 'TERMINALS':
                    break

                current_section = None
                continue

            # Skip EOF marker
            if line.upper() == 'EOF':
                break

            # Process content based on current section
            if current_section == 'GRAPH':
                parts = line.split()
                if not parts:
                    continue

                # Extract nodes and edges counts
                if parts[0].upper() == 'NODES':
                    data['n'] = int(parts[1])
                elif parts[0].upper() == 'EDGES':
                    data['m'] = int(parts[1])
                # Extract edge information (E lines)
                elif parts[0].upper() == 'E':
                    if len(parts) >= 4:  # E u v cost
                        u, v, cost = int(parts[1]), int(parts[2]), float(parts[3])
                        # Store edge with vertices in sorted order
                        data['graph_edges'][(min(u, v), max(u, v))] = cost
                # Skip EA and ED lines (we process but don't store them in the output format)
                elif parts[0].upper() in ['EA', 'ED', 'EC']:
                    continue

            # Extract terminal nodes
            elif current_section == 'TERMINALS':
                parts = line.split()
                if not parts:
                    continue

                # Skip the "Terminals X" line
                if parts[0].upper() == 'TERMINALS':
                    continue

                # Add terminal nodes
                if parts[0].upper() == 'T':
                    if len(parts) >= 2:
                        data['terminals'].append(int(parts[1]))

    return [data]


def eval_func(**kwargs):
    """
    Evaluates the solution for a single test case of the Steiner Tree Problem.

    Parameters:
      case (dict): A dictionary containing the input data with keys:
          - 'n': number of vertices,
          - 'm': number of edges,
          - 'graph_edges': dictionary mapping (min(u,v), max(u,v)) -> cost,
          - 'terminals': list of terminal vertices.
      solution (dict): A dictionary containing the proposed solution with keys:
          - 'declared_cost': the total cost declared by the solution,
          - 'num_edges': number of edges in the solution,
          - 'edges': list of edges (each as a tuple (u, v)).

    Returns:
      float: The computed total cost if the solution is valid, otherwise float('inf').

    The evaluation checks:
      1. Every edge in the solution exists in the input graph.
      2. The sum of the edge costs matches the declared cost (within a small tolerance).
      3. All terminal vertices are connected in the solution subgraph.
    """
    import math

    graph_edges = kwargs['graph_edges']
    terminals = kwargs['terminals']
    declared_cost = kwargs.get('declared_cost', None)
    num_edges = kwargs.get('num_edges', None)
    solution_edges = kwargs.get('edges', None)

    if declared_cost is None or num_edges is None or solution_edges is None:
        raise Exception("Error: The solution must contain 'declared_cost', 'num_edges', and 'edges'.")

    # Check that the number of edges matches.
    if num_edges != len(solution_edges):
        raise Exception("Error: The number of edges declared does not match the number provided.")

    computed_cost = 0.0
    solution_nodes = set()
    for (u, v) in solution_edges:
        key = (min(u, v), max(u, v))
        if key not in graph_edges:
            raise Exception(f"Error: Edge ({u}, {v}) not found in the input graph.")
        computed_cost += graph_edges[key]
        solution_nodes.update([u, v])

    if abs(declared_cost - computed_cost) > 1e-6:
        raise Exception(f"Error: Declared cost ({declared_cost}) does not match computed cost ({computed_cost}).")

    # Check connectivity among terminal vertices using Union-Find.
    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx = find(x)
        ry = find(y)
        if rx != ry:
            parent[ry] = rx

    for node in solution_nodes:
        parent[node] = node
    for (u, v) in solution_edges:
        union(u, v)

    # All terminals must be in the solution and connected.
    for t in terminals:
        if t not in parent:
            raise Exception(f"Error: Terminal node {t} is not present in the solution.")

    root = find(terminals[0])
    for t in terminals:
        if find(t) != root:
            raise Exception("Error: Not all terminal nodes are connected in the solution.")

    return computed_cost


def norm_score(results):
    optimal_scores = {'valid_instances/R25K02EFST.stp': [99.037087811],
                      'valid_instances/R25K05EFST.stp': [99.491232075], 'valid_instances/hc10_valid_1p.stp': [59822.0],
                      'valid_instances/hc9_valid_1p.stp': [30241.0], 'valid_instances/R25K03EFST.stp': [99.215720747],
                      'valid_instances/hc6_valid_1p.stp': [4017.0], 'valid_instances/hc6_valid_0p.stp': [4017.0],
                      'valid_instances/hc10_valid_0p.stp': [59822.0], 'valid_instances/hc9_valid_0p.stp': [30241.0],
                      'valid_instances/hc7_valid_0p.stp': [7922.0], 'valid_instances/hc8_valid_0p.stp': [15331.0],
                      'valid_instances/R25K01EFST.stp': [98.961213426], 'valid_instances/hc8_valid_1p.stp': [15331.0],
                      'valid_instances/R25K04EFST.stp': [98.943139188], 'valid_instances/hc7_valid_1p.stp': [7922.0],
                      'easy_test_instances/G207a.stp': [2265834.0], 'easy_test_instances/G103a.stp': [19938744.0],
                      'easy_test_instances/G304a.stp': [6721180.0], 'easy_test_instances/G305a.stp': [40632152.0],
                      'easy_test_instances/G206a.stp': [9175622.0], 'easy_test_instances/G102a.stp': [15187538.0],
                      'easy_test_instances/G307a.stp': [51219090.0], 'easy_test_instances/G204a.stp': [5313548.0],
                      'easy_test_instances/G101a.stp': [3492405.0], 'easy_test_instances/G205a.stp': [24819583.0],
                      'easy_test_instances/G306a.stp': [33949874.0], 'easy_test_instances/G303a.stp': [27941456.0],
                      'easy_test_instances/G104a.stp': [26165528.0], 'easy_test_instances/G105a.stp': [12507877.0],
                      'easy_test_instances/G201a.stp': [3484028.0], 'easy_test_instances/G302a.stp': [13300990.0],
                      'easy_test_instances/G203a.stp': [13155210.0], 'easy_test_instances/G309a.stp': [11256303.0],
                      'easy_test_instances/G107a.stp': [7325530.0], 'easy_test_instances/G301a.stp': [4797441.0],
                      'easy_test_instances/G202a.stp': [6849423.0], 'easy_test_instances/G308a.stp': [4699474.0],
                      'easy_test_instances/G106a.stp': [44547208.0], 'hard_test_instances/cc6-3p.stp': [20355.0],
                      'hard_test_instances/cc11-2u.stp': [617.0], 'hard_test_instances/bipe2u.stp': [54.0],
                      'hard_test_instances/hc7p.stp': [7905.0], 'hard_test_instances/cc3-10u.stp': [127.0],
                      'hard_test_instances/hc11p.stp': [120000.0], 'hard_test_instances/bip42p.stp': [24657.0],
                      'hard_test_instances/bip52p.stp': [24611.0], 'hard_test_instances/hc10p.stp': [60059.0],
                      'hard_test_instances/cc3-11u.stp': [154.0], 'hard_test_instances/bipa2p.stp': [35393.0],
                      'hard_test_instances/cc6-2p.stp': [3271.0], 'hard_test_instances/cc5-3u.stp': [71.0],
                      'hard_test_instances/hc6p.stp': [4003.0], 'hard_test_instances/cc12-2p.stp': [122007.0],
                      'hard_test_instances/cc10-2u.stp': [344.0], 'hard_test_instances/hc8u.stp': [148.0],
                      'hard_test_instances/cc7-3p.stp': [57638.0], 'hard_test_instances/cc3-5p.stp': [3661.0],
                      'hard_test_instances/hc12p.stp': [238889.0], 'hard_test_instances/cc9-2u.stp': [169.0],
                      'hard_test_instances/bip62u.stp': [220.0], 'hard_test_instances/hc9u.stp': [292.0],
                      'hard_test_instances/cc3-4p.stp': [2338.0], 'hard_test_instances/cc3-12u.stp': [187.0],
                      'hard_test_instances/hc9p.stp': [30254.0], 'hard_test_instances/cc3-12p.stp': [18928.0],
                      'hard_test_instances/cc3-4u.stp': [23.0], 'hard_test_instances/bip62p.stp': [22851.0],
                      'hard_test_instances/cc9-2p.stp': [17285.0], 'hard_test_instances/cc7-3u.stp': [556.0],
                      'hard_test_instances/hc8p.stp': [15322.0], 'hard_test_instances/cc10-2p.stp': [35514.0],
                      'hard_test_instances/hc12u.stp': [2330.0], 'hard_test_instances/cc3-5u.stp': [36.0],
                      'hard_test_instances/cc6-2u.stp': [32.0], 'hard_test_instances/cc12-2u.stp': [1185.0],
                      'hard_test_instances/hc6u.stp': [39.0], 'hard_test_instances/cc5-3p.stp': [7299.0],
                      'hard_test_instances/bip52u.stp': [234.0], 'hard_test_instances/bip42u.stp': [236.0],
                      'hard_test_instances/bipa2u.stp': [340.0], 'hard_test_instances/cc3-11p.stp': [15674.0],
                      'hard_test_instances/hc10u.stp': [575.0], 'hard_test_instances/hc11u.stp': [1159.0],
                      'hard_test_instances/cc3-10p.stp': [12866.0], 'hard_test_instances/cc11-2p.stp': [63608.0],
                      'hard_test_instances/cc6-3u.stp': [200.0], 'hard_test_instances/hc7u.stp': [77.0],
                      'hard_test_instances/bipe2p.stp': [5616.0]}

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
