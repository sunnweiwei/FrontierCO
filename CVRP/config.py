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
    
    return {
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
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    x1, y1 = coordinates[i]
                    x2, y2 = coordinates[j]
                    # Euclidean distance
                    dist_matrix[i, j] = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    return n, capacity, coordinates, demands, dist_matrix, depot_idx


def eval_func(nodes, demands, capacity, depot_idx, optimal_routes, predicted_routes):
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
    validate_cvrp_solution(nodes, demands, capacity, depot_idx, predicted_routes)
    
    # Calculate the predicted solution cost (total distance)
    pred_cost = calculate_total_distance(nodes, predicted_routes)
    
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