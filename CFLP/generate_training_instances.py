import numpy as np
import random
import argparse
import os

def generate_capacitated_facility_location_instance(random, n_customers, n_facilities, ratio):
    """
    Generate a Capacitated Facility Location problem following
        Cornuejols G, Sridharan R, Thizy J-M (1991)
        A Comparison of Heuristics and Relaxations for the Capacitated Plant Location Problem.
        European Journal of Operations Research 50:280-297.
    
    Parameters
    ----------
    random : numpy.random.RandomState
        A random number generator.
    n_customers: int
        The desired number of customers.
    n_facilities: int
        The desired number of facilities.
    ratio: float
        The desired capacity / demand ratio.
        
    Returns
    -------
    capacities : numpy.ndarray
        Capacities for each facility.
    fixed_costs : numpy.ndarray
        Fixed costs for each facility.
    demands : numpy.ndarray
        Demands for each customer.
    trans_costs : numpy.ndarray
        Transportation costs between each customer and facility.
    """
    # Generate random coordinates for customers and facilities
    c_x = random.rand(n_customers)
    c_y = random.rand(n_customers)
    f_x = random.rand(n_facilities)
    f_y = random.rand(n_facilities)
    
    # Generate random demands and capacities
    demands = random.randint(5, 35+1, size=n_customers)
    capacities = random.randint(10, 160+1, size=n_facilities)
    
    # Calculate fixed costs
    fixed_costs = random.randint(100, 110+1, size=n_facilities) * np.sqrt(capacities) \
            + random.randint(90+1, size=n_facilities)
    fixed_costs = fixed_costs.astype(int)
    
    # Adjust capacities according to ratio
    total_demand = demands.sum()
    total_capacity = capacities.sum()
    capacities = capacities * ratio * total_demand / total_capacity
    capacities = capacities.astype(int)
    
    # Calculate transportation costs
    trans_costs = np.sqrt(
            (c_x.reshape((-1, 1)) - f_x.reshape((1, -1))) ** 2 \
            + (c_y.reshape((-1, 1)) - f_y.reshape((1, -1))) ** 2) * 10 * demands.reshape((-1, 1))
    trans_costs = trans_costs.astype(int)
    
    return capacities, fixed_costs, demands, trans_costs

def generate_beasley_instances(random, n_customers, n_facilities, desired_open_facilities):
    """
    Generate a Capacitated Facility Location problem following Beasley's approach.
    
    Parameters
    ----------
    random : numpy.random.RandomState
        A random number generator.
    n_customers: int
        The desired number of customers.
    n_facilities: int
        The desired number of facilities.
    desired_open_facilities: int
        The desired number of open facilities in the optimal solution (P0).
    
    Problem types from Beasley's paper:
    ----------------------------------
    Type A: n=100, m=1000, capacities=[8000,10000,12000,14000], P0=5
    Type B: n=100, m=1000, capacities=[5000,6000,7000,8000], P0=10 
    Type C: n=100, m=1000, capacities=[5000,5750,6500,7250], P0=15
    Type D: n=500, m=1000, capacities=[7000,8000,9000,10000], P0=5
    Type E: n=500, m=1000, capacities=[5000,6000,7000,8000], P0=10
        
    Returns
    -------
    capacities : numpy.ndarray
        Capacities for each facility (all identical).
    fixed_costs : numpy.ndarray
        Fixed costs for each facility.
    demands : numpy.ndarray
        Demands for each customer.
    trans_costs : numpy.ndarray
        Transportation costs between each customer and facility.
    """
    # Generate random coordinates in a 1000x1000 Euclidean square
    c_x = random.rand(n_customers) * 1000
    c_y = random.rand(n_customers) * 1000
    f_x = random.rand(n_facilities) * 1000
    f_y = random.rand(n_facilities) * 1000
    
    # Generate random demands in range [1, 100]
    demands = random.randint(1, 100+1, size=n_customers)
    
    # Calculate Euclidean distances
    distances = np.sqrt(
            (c_x.reshape((-1, 1)) - f_x.reshape((1, -1))) ** 2 \
            + (c_y.reshape((-1, 1)) - f_y.reshape((1, -1))) ** 2)
    
    # Calculate transportation costs: distance × demand × random factor in [1.00, 1.25]
    random_factors = random.uniform(1.00, 1.25, size=(n_customers, n_facilities))
    trans_costs = distances * demands.reshape((-1, 1)) * random_factors
    trans_costs = trans_costs.astype(int)
    
    # Set the same capacity for all facilities (from Table 1 in the paper)
    # Note: Adapt this based on your specific problem class (A, B, C, D, E)
    # This is a placeholder - set the appropriate capacity based on your needs
    capacity = 8000  # Example value - adjust as needed
    capacities = np.full(n_facilities, capacity, dtype=int)
    
    # Generate fixed costs using the method described in the paper
    # Step 1: Randomly select (P0 + 1) warehouses
    selected_warehouses = random.choice(n_facilities, desired_open_facilities + 1, replace=False)
    
    # Step 2: Calculate cost D1 by assigning each customer to its cheapest warehouse
    # Create a mask for the selected warehouses
    mask = np.zeros(n_facilities, dtype=bool)
    mask[selected_warehouses] = True
    
    # Find cheapest warehouse for each customer within the selected set
    cheapest_costs = np.min(np.where(mask, trans_costs, np.inf), axis=1)
    D1 = np.sum(cheapest_costs)
    
    # Step 3: Remove two random warehouses and calculate D2
    removed_warehouses = random.choice(selected_warehouses, 2, replace=False)
    mask[removed_warehouses] = False
    
    # Find cheapest warehouse for each customer within the reduced set
    cheapest_costs = np.min(np.where(mask, trans_costs, np.inf), axis=1)
    D2 = np.sum(cheapest_costs)
    
    # Step 4: Set fixed costs using the formula (D2 - D1)/2 × random factor in [0.75, 1.25]
    fixed_cost_base = (D2 - D1) / 2
    random_factors = random.uniform(0.75, 1.25, size=n_facilities)
    fixed_costs = np.ones(n_facilities) * fixed_cost_base * random_factors
    fixed_costs = fixed_costs.astype(int)
    
    return capacities, fixed_costs, demands, trans_costs

def save_cflp_instance(filename, n_facilities, n_customers, capacities, fixed_costs, demands, trans_costs):
    """
    Save a Capacitated Facility Location Problem instance to a file.
    
    Parameters
    ----------
    filename : str
        Path to the file to save.
    n_facilities : int
        Number of facilities.
    n_customers : int
        Number of customers.
    capacities : numpy.ndarray
        Capacities for each facility.
    fixed_costs : numpy.ndarray
        Fixed costs for each facility.
    demands : numpy.ndarray
        Demands for each customer.
    trans_costs : numpy.ndarray
        Transportation costs between each customer and facility.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
    
    with open(filename, 'w') as f:
        # First line: number of potential warehouse locations (m), number of customers (n)
        f.write(f"{n_facilities} {n_customers}\n")
        
        # For each potential warehouse location: capacity, fixed cost
        for i in range(n_facilities):
            f.write(f"{capacities[i]} {fixed_costs[i]}\n")
        
        line = " ".join([str(demands[i])  for i in range(n_customers)])
        f.write(line + "\n")

        # For each customer: demand, cost of allocating all demand to each warehouse
        for j in range(n_facilities):
            line = " ".join([str(trans_costs[i, j]) for i in range(n_customers)])
            f.write(line + "\n")
    
    return filename

def main():
    parser = argparse.ArgumentParser(description='Generate test instances for the Capacitated Facility Location Problem')
    
    # Problem size parameters
    parser.add_argument('--customers', type=int, nargs='+', default=[100],
                        help='List of customer counts (default: [100])')
    parser.add_argument('--facilities', type=int, nargs='+', default=[100],
                        help='List of facility counts (default: [100])')
    
    # Method parameters
    parser.add_argument('--method', type=str, choices=['cornuejols', 'beasley'], default='cornuejols',
                        help='Method to generate instances (default: cornuejols)')
    
    # Ratio parameter (for Cornuejols method)
    parser.add_argument('--ratio', type=float, nargs='+', default=[5.0],
                        help='List of capacity/demand ratios for Cornuejols method (default: [5.0])')
    
    # Desired open facilities parameter (for Beasley method)
    parser.add_argument('--open_facilities', type=int, nargs='+', default=[5],
                        help='List of desired number of open facilities for Beasley method (default: [5])')
    
    # Problem type for Beasley method (A, B, C, D, E)
    parser.add_argument('--problem_type', type=str, choices=['A', 'B', 'C', 'D', 'E'], 
                        help='Predefined problem type for Beasley method')
    
    # Facility capacities (for Beasley method)
    parser.add_argument('--facility_capacity', type=int, nargs='+', default=[8000],
                        help='List of facility capacities for Beasley method (default: [8000])')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--output_dir', type=str, default='valid_instances',
                        help='Directory to save generated instances (default: valid_instances)')
    parser.add_argument('--instances_per_config', type=int, default=1,
                        help='Number of instances to generate per configuration (default: 1)')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Generate instances
    total_instances = 0
    
    for n_customers in args.customers:
        for n_facilities in args.facilities:
            if args.method == 'cornuejols':
                for ratio in args.ratio:
                    for instance_num in range(1, args.instances_per_config + 1):
                        # Set a different seed for each instance
                        instance_seed = args.seed + total_instances
                        rng = np.random.RandomState(instance_seed)
                        
                        # Create filename
                        if args.instances_per_config > 1:
                            filename = f"{args.output_dir}/cflp_corn_n{n_customers}_m{n_facilities}_r{ratio:.1f}_{instance_num}.txt"
                        else:
                            filename = f"{args.output_dir}/cflp_corn_n{n_customers}_m{n_facilities}_r{ratio:.1f}.txt"
                        
                        # Generate the instance
                        capacities, fixed_costs, demands, trans_costs = generate_capacitated_facility_location_instance(
                            rng, n_customers, n_facilities, ratio
                        )
                        
                        # Save the instance
                        save_cflp_instance(
                            filename, n_facilities, n_customers,
                            capacities, fixed_costs, demands, trans_costs
                        )
                        
                        print(f"Generated instance: {os.path.basename(filename)}")
                        total_instances += 1
            
            elif args.method == 'beasley':
                # If problem type is specified, use predefined parameters
                if args.problem_type:
                    # Define parameters for each problem type
                    problem_params = {
                        'A': {'n': 100, 'm': 1000, 'capacities': [8000, 10000, 12000, 14000], 'p0': 5},
                        'B': {'n': 100, 'm': 1000, 'capacities': [5000, 6000, 7000, 8000], 'p0': 10},
                        'C': {'n': 100, 'm': 1000, 'capacities': [5000, 5750, 6500, 7250], 'p0': 15},
                        'D': {'n': 500, 'm': 1000, 'capacities': [7000, 8000, 9000, 10000], 'p0': 5},
                        'E': {'n': 500, 'm': 1000, 'capacities': [5000, 6000, 7000, 8000], 'p0': 10}
                    }
                    
                    params = problem_params[args.problem_type]
                    n_customers = params['n']
                    n_facilities = params['m']
                    capacities_list = params['capacities']
                    open_facilities_list = [params['p0']]
                    
                    print(f"Using predefined parameters for problem type {args.problem_type}:")
                    print(f"  Customers: {n_customers}")
                    print(f"  Facilities: {n_facilities}")
                    print(f"  Capacities: {capacities_list}")
                    print(f"  Desired open facilities: {open_facilities_list[0]}")
                    
                    for capacity in capacities_list:
                        for instance_num in range(1, args.instances_per_config + 1):
                            # Set a different seed for each instance
                            instance_seed = args.seed + total_instances
                            rng = np.random.RandomState(instance_seed)
                            
                            # Create filename
                            if args.instances_per_config > 1:
                                filename = f"{args.output_dir}/cflp_beasley_type{args.problem_type}_c{capacity}_{instance_num}.txt"
                            else:
                                filename = f"{args.output_dir}/cflp_beasley_type{args.problem_type}_c{capacity}.txt"
                            
                            # Generate the instance
                            capacities, fixed_costs, demands, trans_costs = generate_beasley_instances(
                                rng, n_customers, n_facilities, open_facilities_list[0]
                            )
                            
                            # Override capacities with the specified value
                            capacities = np.full(n_facilities, capacity, dtype=int)
                            
                            # Save the instance
                            save_cflp_instance(
                                filename, n_facilities, n_customers,
                                capacities, fixed_costs, demands, trans_costs
                            )
                            
                            print(f"Generated instance: {os.path.basename(filename)}")
                            total_instances += 1
                else:
                    # Use command line parameters if problem type is not specified
                    for open_facilities in args.open_facilities:
                        for capacity in args.facility_capacity:
                            for instance_num in range(1, args.instances_per_config + 1):
                                # Set a different seed for each instance
                                instance_seed = args.seed + total_instances
                                rng = np.random.RandomState(instance_seed)
                                
                                # Create filename
                                if args.instances_per_config > 1:
                                    filename = f"{args.output_dir}/cflp_beasley_n{n_customers}_m{n_facilities}_p{open_facilities}_c{capacity}_{instance_num}.txt"
                                else:
                                    filename = f"{args.output_dir}/cflp_beasley_n{n_customers}_m{n_facilities}_p{open_facilities}_c{capacity}.txt"
                                
                                # Generate the instance with override for capacity
                                capacities, fixed_costs, demands, trans_costs = generate_beasley_instances(
                                    rng, n_customers, n_facilities, open_facilities
                                )
                                
                                # Override capacities with the specified value
                                capacities = np.full(n_facilities, capacity, dtype=int)
                                
                                # Save the instance
                                save_cflp_instance(
                                    filename, n_facilities, n_customers,
                                    capacities, fixed_costs, demands, trans_costs
                                )
                                
                                print(f"Generated instance: {os.path.basename(filename)}")
                                total_instances += 1
                            
                            print(f"Generated instance: {os.path.basename(filename)}")
                            total_instances += 1
    
    print(f"\nTotal instances generated: {total_instances}")
    print(f"Instances saved in directory: {args.output_dir}")

if __name__ == "__main__":
    main()