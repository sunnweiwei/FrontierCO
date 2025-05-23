import numpy as np
import os
import argparse

def generate_p_median_instance(n_customers, n_facilities, 
                              demand_range=(1, 20), 
                              tightness_range=(0.82, 0.96),
                              seed=42):
    """
    Generate a capacitated p-median problem instance.
    
    Parameters
    ----------
    n_customers: int
        The number of customers.
    n_facilities: int
        The number of facility locations (p).
    demand_range: tuple
        The range for demand values (min, max).
    tightness_range: tuple
        The range for tightness ratio.
    seed: int
        Random seed for reproducibility.
        
    Returns
    -------
    coordinates : numpy.ndarray
        Coordinates for each customer.
    demands : numpy.ndarray
        Demands for each customer.
    capacity : int
        Uniform capacity for each facility.
    """
    # Set random seed
    np.random.seed(seed)
    
    # Generate random coordinates in the range [1, 100] as integers
    coordinates = np.random.randint(1, 101, size=(n_customers, 2))
    
    # Generate random demands in the specified range as integers
    demands = np.random.randint(demand_range[0], demand_range[1] + 1, size=n_customers)
    
    # Determine a tightness ratio within the specified range
    tightness = np.random.uniform(tightness_range[0], tightness_range[1])
    
    # Calculate total demand
    total_demand = np.sum(demands)
    
    # Calculate the required total capacity
    required_total_capacity = total_demand / tightness
    
    # Calculate the constant capacity for each facility
    capacity = int(np.ceil(required_total_capacity / n_facilities))
    
    # Verify the tightness ratio
    actual_tightness = total_demand / (capacity * n_facilities)
    print(f"Target tightness: {tightness:.4f}, Actual tightness: {actual_tightness:.4f}")
    
    return coordinates, demands, capacity

def save_instance(filename, coordinates, demands, capacity, n_facilities):
    """
    Save a p-median problem instance to a file.
    
    Format:
    - First line: number of points (n) number of medians (p)
    - Other lines: X Y coordinates, capacity, demand for each point
    """
    n_customers = len(coordinates)
    
    with open(filename, 'w') as f:
        # First line: number of points (n) number of medians (p)
        f.write(f"{n_customers} {n_facilities}\n")
        
        # Write coordinates, capacity, and demand for each point
        for i in range(n_customers):
            f.write(f"{coordinates[i, 0]} {coordinates[i, 1]} {capacity} {demands[i]}\n")
    
    return filename

def main():
    parser = argparse.ArgumentParser(description='Generate p-median problem instances')
    
    # Problem size parameters
    parser.add_argument('--n', type=int, nargs='+', default=[50, 100], 
                        help='Customer counts (default: [50, 100])')
    parser.add_argument('--p', type=int, nargs='+', default=[5, 10],
                        help='Facility counts (default: [5, 10])')
    
    # Other parameters
    parser.add_argument('--demand_min', type=int, default=1,
                        help='Minimum demand value (default: 1)')
    parser.add_argument('--demand_max', type=int, default=20,
                        help='Maximum demand value (default: 20)')
    parser.add_argument('--tightness_min', type=float, default=0.82,
                        help='Minimum tightness ratio (default: 0.82)')
    parser.add_argument('--tightness_max', type=float, default=0.96,
                        help='Maximum tightness ratio (default: 0.96)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--output_dir', type=str, default='training_instances',
                        help='Output directory (default: training_instances)')
    parser.add_argument('--instances', type=int, default=10,
                        help='Instances per configuration (default: 10)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    total_instances = 0
    
    for n in args.n:
        for p in args.p:
            for i in range(1, args.instances + 1):
                # Unique seed for each instance
                instance_seed = args.seed + total_instances
                
                # Generate instance
                coordinates, demands, capacity = generate_p_median_instance(
                    n, p,
                    demand_range=(args.demand_min, args.demand_max),
                    tightness_range=(args.tightness_min, args.tightness_max),
                    seed=instance_seed
                )
                
                # Create filenames
                base_filename = f"p_median_n{n}_p{p}_{i}"
                txt_filename = f"{args.output_dir}/{base_filename}.txt"
                lp_filename = f"{args.output_dir}/{base_filename}.lp"
                
                 # Save instance in text format
                save_instance(txt_filename, coordinates, demands, capacity, p)


                print(f"Generated: {base_filename}.txt")
                print(f"  Customers: {n}, Facilities: {p}, Capacity: {capacity}")
                
                total_instances += 1
    
    print(f"\nGenerated {total_instances} instances in {args.output_dir}")

if __name__ == "__main__":
    main()