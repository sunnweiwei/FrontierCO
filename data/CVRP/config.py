DESCRIPTION = '''The Capacitated Vehicle Routing Problem (CVRP) is a classic optimization problem that extends the Traveling Salesman Problem. In the CVRP, a fleet of vehicles with limited capacity must service a set of customers with specific demands, starting and ending at a central depot. Each customer must be visited exactly once by exactly one vehicle, and the total demand of customers on a single vehicle's route cannot exceed the vehicle's capacity. The objective is to minimize the total travel distance while satisfying all customer demands and vehicle capacity constraints.'''

import numpy as np
import math


def solve(**kwargs):
    """
    Solve a CVRP instance.

    Args:
        - nodes (list): List of (x, y) coordinates representing locations (depot and customers)
                     Format: [(x1, y1), (x2, y2), ..., (xn, yn)]
        - demands (list): List of customer demands, where demands[i] is the demand for customer i
                     Format: [d0, d1, d2, ..., dn]
        - capacity (int): Vehicle capacity
        - depot_idx (int): Index of the depot in the nodes list (typically 0)

    Returns:
        dict: Solution information with:
            - 'routes' (list): List of routes, where each route is a list of node indices
                            Format: [[0, 3, 1, 0], [0, 2, 5, 0], ...] where 0 is the depot
    """

    # This is a placeholder implementation
    # Your solver implementation would go here
    # Your function must yield multiple solutions over time, not just return one solution
    # Use Python's yield keyword repeatedly to produce a stream of solutions
    # Each yielded solution should be better than the previous one
    while True:
        yield {
            'routes': [],
        }


def load_data(file_path):
    """
    Load CVRP instances from .vrp files.

    Args:
        file_path (str): Path to the file containing CVRP instances

    Returns:
        list: List of dictionaries, each containing a CVRP instance with:
            - 'nodes': List of (x, y) coordinates
            - 'demands': List of customer demands
            - 'capacity': Vehicle capacity
            - 'depot_idx': Index of the depot (typically 0)
            - 'optimal_routes': List of optimal routes (if available)
    """
    instances = []

    try:
        n, capacity, coordinates, demands, dist_matrix, depot_idx = read_vrp_file(file_path)

        # Create a dictionary for this instance
        instance = {
            'nodes': coordinates,
            'demands': demands,
            'capacity': capacity,
            'depot_idx': depot_idx,
            'dist_matrix': dist_matrix,
            'optimal_routes': None  # Typically not available in .vrp files
        }

        instances.append(instance)
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

    return instances


def read_vrp_file(filename):
    """Read CVRP Problem instance from a .vrp file."""
    with open(filename, 'r') as f:
        lines = f.readlines()

        # Parse metadata from header
        n = None  # Number of nodes (including depot)
        capacity = None
        coordinates = []
        demands = []
        depot_idx = 0  # Default depot index - node 1 (0-indexed) is standard default
        depot_found = False

        # Read through file sections
        section = None
        for line in lines:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Check for section headers
            if line.startswith("DIMENSION"):
                n = int(line.split(":")[1].strip())
            elif line.startswith("CAPACITY"):
                capacity = int(line.split(":")[1].strip())
            elif line == "NODE_COORD_SECTION":
                section = "coords"
                continue
            elif line == "DEMAND_SECTION":
                section = "demand"
                continue
            elif line == "DEPOT_SECTION":
                section = "depot"
                depot_found = True
                continue
            elif line == "EOF":
                break

            # Parse data based on current section
            if section == "coords":
                parts = line.split()
                if len(parts) >= 3:
                    node_id = int(parts[0]) - 1  # Convert to 0-indexed
                    x = float(parts[1])
                    y = float(parts[2])

                    # Ensure we have a spot in the array for this node
                    while len(coordinates) <= node_id:
                        coordinates.append(None)

                    coordinates[node_id] = (x, y)

            elif section == "demand":
                parts = line.split()
                if len(parts) >= 2:
                    node_id = int(parts[0]) - 1  # Convert to 0-indexed
                    demand = int(parts[1])

                    # Ensure we have a spot in the array for this demand
                    while len(demands) <= node_id:
                        demands.append(None)

                    demands[node_id] = demand

            elif section == "depot":
                try:
                    depot = int(line)
                    if depot > 0:  # Valid depot ID
                        depot_idx = depot - 1  # Convert to 0-indexed
                except ValueError:
                    pass  # Skip if not a valid depot ID

        # Ensure all node coordinates and demands are loaded
        if n is None:
            n = len(coordinates)

        # Handle special cases for depot identification
        if not depot_found:
            # Check if there's a node with zero demand - that's often the depot
            for i, demand in enumerate(demands):
                if demand == 0:
                    depot_idx = i
                    break
            # Otherwise, use node 0 as default depot (common convention)

        # Calculate distance matrix
        coords = np.array(coordinates)

        # Calculate pairwise squared differences
        x_diff = coords[:, 0, np.newaxis] - coords[:, 0]
        y_diff = coords[:, 1, np.newaxis] - coords[:, 1]

        # Calculate Euclidean distances
        dist_matrix = np.sqrt(x_diff ** 2 + y_diff ** 2)

        # Set diagonal to zero (optional, as it should already be zero)
        np.fill_diagonal(dist_matrix, 0)

    return n, capacity, coordinates, demands, dist_matrix, depot_idx


def eval_func(nodes, demands, capacity, depot_idx, optimal_routes, routes, **kwargs):
    """
    Evaluate a predicted CVRP solution against optimal routes or calculate total distance.

    Args:
        nodes (list): List of (x, y) coordinates representing locations
                    Format: [(x1, y1), (x2, y2), ..., (xn, yn)]
        demands (list): List of customer demands
                    Format: [d0, d1, d2, ..., dn]
        capacity (int): Vehicle capacity
        depot_idx (int): Index of the depot (typically 0)
        optimal_routes (list): Reference optimal routes (may be None if not available)
                            Format: [[0, 3, 1, 0], [0, 2, 5, 0], ...]
        predicted_routes (list): Predicted routes from the solver
                              Format: [[0, 3, 1, 0], [0, 2, 5, 0], ...]

    Returns:
        float: Optimality gap percentage if optimal_routes is provided,
              or just the predicted solution's total distance
    """
    # Validate solution
    validate_cvrp_solution(nodes, demands, capacity, depot_idx, routes)

    # Calculate the predicted solution cost (total distance)
    pred_cost = calculate_total_distance(nodes, routes)

    # If optimal routes are provided, calculate optimality gap
    if optimal_routes:
        opt_cost = calculate_total_distance(nodes, optimal_routes)
        opt_gap = ((pred_cost / opt_cost) - 1) * 100
        return opt_gap

    # Otherwise, just return the predicted cost
    return pred_cost


def validate_cvrp_solution(nodes, demands, capacity, depot_idx, routes):
    """
    Validate that a CVRP solution meets all constraints.

    Args:
        nodes (list): List of (x, y) coordinates
        demands (list): List of customer demands
        capacity (int): Vehicle capacity
        depot_idx (int): Index of the depot
        routes (list): List of routes to validate

    Raises:
        Exception: If the solution is invalid
    """
    num_nodes = len(nodes)
    all_visited = set()

    for route_idx, route in enumerate(routes):
        # Check that route starts and ends at depot
        if route[0] != depot_idx or route[-1] != depot_idx:
            raise Exception(f"Route {route_idx} does not start and end at the depot")

        # Check capacity constraint
        route_demand = sum(demands[i] for i in route[1:-1])  # Exclude depot
        if route_demand > capacity:
            raise Exception(f"Route {route_idx} exceeds capacity: {route_demand} > {capacity}")

        # Check that nodes are valid indices
        for node in route:
            if node < 0 or node >= num_nodes:
                raise Exception(f"Invalid node index {node} in route {route_idx}")

            # Add to visited set (excluding depot)
            if node != depot_idx:
                all_visited.add(node)

    # Check that all customers are visited exactly once
    expected_visited = set(range(num_nodes))
    expected_visited.remove(depot_idx)  # Exclude depot

    if all_visited != expected_visited:
        missing = expected_visited - all_visited
        duplicate = all_visited - expected_visited

        if missing:
            raise Exception(f"Nodes not visited: {missing}")
        if duplicate:
            raise Exception(f"Nodes visited more than once: {duplicate}")


def calculate_total_distance(nodes, routes):
    """
    Calculate the total distance of a CVRP solution.

    Args:
        nodes (list): List of (x, y) coordinates
        routes (list): List of routes

    Returns:
        float: Total distance of all routes
    """
    total_distance = 0

    for route in routes:
        route_distance = 0
        for i in range(len(route) - 1):
            from_node = route[i]
            to_node = route[i + 1]

            # Calculate Euclidean distance
            from_x, from_y = nodes[from_node]
            to_x, to_y = nodes[to_node]
            segment_distance = math.sqrt((to_x - from_x) ** 2 + (to_y - from_y) ** 2)

            route_distance += segment_distance

        total_distance += route_distance

    return total_distance

def norm_score(results):
    optimal_scores = {'easy_test_instances/Golden_13.vrp': [857.188745], 'easy_test_instances/Golden_17.vrp': [707.755935], 'easy_test_instances/Golden_10.vrp': [735.43], 'easy_test_instances/Golden_19.vrp': [1365.6], 'easy_test_instances/Golden_7.vrp': [10023.844627], 'easy_test_instances/Golden_3.vrp': [10785.779388], 'easy_test_instances/Golden_1.vrp': [5370.545835], 'easy_test_instances/Golden_8.vrp': [11486.585777], 'easy_test_instances/Golden_12.vrp': [1100.67], 'easy_test_instances/Golden_5.vrp': [6460.979519], 'easy_test_instances/Golden_18.vrp': [995.13], 'easy_test_instances/Golden_9.vrp': [579.7], 'easy_test_instances/Golden_11.vrp': [911.98], 'easy_test_instances/Golden_4.vrp': [13541.657456], 'easy_test_instances/Golden_16.vrp': [1611.28], 'easy_test_instances/Golden_6.vrp': [8348.949187], 'easy_test_instances/Golden_15.vrp': [1337.27], 'easy_test_instances/Golden_2.vrp': [8205.866802], 'easy_test_instances/Golden_20.vrp': [1817.59], 'easy_test_instances/Golden_14.vrp': [1080.55], 'hard_test_instances/Leuven1.vrp': [192848.0], 'hard_test_instances/Leuven2.vrp': [111391.0], 'hard_test_instances/Antwerp1.vrp': [477277.0], 'hard_test_instances/Antwerp2.vrp': [291350.0], 'hard_test_instances/Ghent1.vrp': [469531.0], 'hard_test_instances/Ghent2.vrp': [257748.0], 'hard_test_instances/Brussels1.vrp': [501719.0], 'hard_test_instances/Brussels2.vrp': [345468.0], 'hard_test_instances/Flanders1.vrp': [7240118.0], 'hard_test_instances/Flanders2.vrp': [4373244.0], 'valid_instances/instance_100_5.vrp': [1269.090891], 'valid_instances/instance_100_1.vrp': [1322.220351], 'valid_instances/instance_100_2.vrp': [1273.835052], 'valid_instances/instance_100_3.vrp': [1239.690157], 'valid_instances/instance_100_4.vrp': [1289.129098], 'valid_instances/instance_50_1.vrp': [753.946825], 'valid_instances/instance_50_3.vrp': [826.843367], 'valid_instances/instance_50_5.vrp': [818.44171], 'valid_instances/instance_50_4.vrp': [776.674785], 'valid_instances/instance_20_1.vrp': [453.252768], 'valid_instances/instance_50_2.vrp': [743.427813], 'valid_instances/instance_20_3.vrp': [487.1528], 'valid_instances/instance_20_4.vrp': [463.300497], 'valid_instances/instance_20_5.vrp': [388.950037], 'valid_instances/instance_20_2.vrp': [455.784836]}

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
