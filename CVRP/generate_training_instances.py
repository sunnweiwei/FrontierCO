import numpy as np
import argparse
import os
import math

def gen_distance_matrix(locations):
    """Generate a distance matrix based on Euclidean distances between locations."""
    n = locations.shape[0]
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distances[i, j] = np.sqrt(np.sum((locations[i] - locations[j]) ** 2))
    return distances

def gen_instance(n, capacity=50, demand_low=1, demand_high=9, depot_coor=(0.5, 0.5)):
    """Generate a CVRP instance."""
    locations = np.random.rand(n, 2)
    demands = np.random.randint(low=demand_low, high=demand_high+1, size=n)
    depot = np.array([depot_coor])
    all_locations = np.vstack((depot, locations))
    all_demands = np.concatenate((np.zeros(1), demands))
    distances = gen_distance_matrix(all_locations)
    return all_locations, all_demands, distances  # (n+1, 2), (n+1), (n+1, n+1)

def write_vrp_file(filename, locations, demands, capacity, comment=""):
    """Write a CVRP instance to a .vrp file in the TSPLIB format."""
    n = locations.shape[0]
    
    with open(filename, 'w') as f:
        # Write header
        instance_name = os.path.basename(filename).split('.')[0]
        f.write(f"NAME : {instance_name}\n")
        f.write(f"COMMENT : {comment}\n")
        f.write("TYPE : CVRP\n")
        f.write(f"DIMENSION : {n}\n")
        f.write("EDGE_WEIGHT_TYPE : EUC_2D\n")
        f.write(f"CAPACITY : {capacity}\n")
        
        # Write node coordinates
        f.write("NODE_COORD_SECTION\n")
        for i in range(n):
            x, y = locations[i]
            # Scale coordinates to avoid too many decimal places
            x, y = x * 100, y * 100
            f.write(f"{i+1} {x:.5f} {y:.5f}\n")
        
        # Write demand section
        f.write("DEMAND_SECTION\n")
        for i in range(n):
            f.write(f"{i+1} {int(demands[i])}\n")
        
        # Write depot section
        f.write("DEPOT_SECTION\n")
        f.write("1\n")  # Depot is always the first node in our format
        f.write("-1\n")
        
        # End the file
        f.write("EOF\n")

def main():
    parser = argparse.ArgumentParser(description='Generate CVRP instances')
    parser.add_argument('--sizes', type=int, nargs='+', default=[20, 50, 100],
                        help='Sizes of instances to generate (default: 20 50 100)')
    parser.add_argument('--instances', type=int, default=10,
                        help='Instances per configuration (default: 10)')
    parser.add_argument('--capacity', type=int, default=50,
                        help='Vehicle capacity (default: 50)')
    parser.add_argument('--demand_low', type=int, default=1,
                        help='Lower bound for demands (default: 1)')
    parser.add_argument('--demand_high', type=int, default=9,
                        help='Upper bound for demands (default: 9)')
    parser.add_argument('--depot_x', type=float, default=0.5,
                        help='Depot x-coordinate (default: 0.5)')
    parser.add_argument('--depot_y', type=float, default=0.5,
                        help='Depot y-coordinate (default: 0.5)')
    parser.add_argument('--output', type=str, default='instances',
                        help='Output directory (default: instances)')
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
    
    # Generate instances for each size
    print(f"Generating instances in {args.output} directory...")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    for size in args.sizes:
        print(f"Generating {args.instances} instances with {size} customers...")
        for i in range(args.instances):
            # Generate instance
            locations, demands, distances = gen_instance(
                n=size, 
                capacity=args.capacity,
                demand_low=args.demand_low,
                demand_high=args.demand_high,
                depot_coor=(args.depot_x, args.depot_y)
            )
            
            
            # Write instance to file
            filename = os.path.join(args.output, f"instance_{size}_{i+1}.vrp")
            write_vrp_file(
                filename=filename,
                locations=locations,
                demands=demands,
                capacity=args.capacity,
                comment=f"Generated instance with {size} customers"
            )
            
        print(f"Done generating instances for size {size}")
    
    print(f"All instances generated in {args.output} directory")

if __name__ == "__main__":
    main()