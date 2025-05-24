import numpy as np
import argparse
import os
import math

def write_tsp_file(filename, locations, comment=""):
    """Write a TSP instance to a .tsp file in the TSPLIB format."""
    n = locations.shape[0]
    
    with open(filename, 'w') as f:
        # Write header
        instance_name = os.path.basename(filename).split('.')[0]
        f.write(f"NAME : {instance_name}\n")
        f.write(f"COMMENT : {comment}\n")
        f.write("TYPE : TSP\n")
        f.write(f"DIMENSION : {n}\n")
        f.write("EDGE_WEIGHT_TYPE : EUC_2D\n")
        
        # Write node coordinates
        f.write("NODE_COORD_SECTION\n")
        for i in range(n):
            x, y = locations[i]
            # Scale coordinates to avoid too many decimal places and match your format
            x, y = x * 1000000, y * 1000000
            f.write(f"{i+1} {int(x)} {int(y)}\n")
        
        # End the file
        f.write("EOF\n")

def generate_batch_instances(batch_size, min_nodes, max_nodes, output_dir, seed_start=1):
    """Generate a batch of TSP instances with varying sizes."""
    instances_generated = 0
    
    for batch_idx in range(batch_size):
        # Sample number of nodes for this instance
        num_nodes = np.random.randint(low=min_nodes, high=max_nodes + 1)
        
        # Generate coordinates uniformly from unit square
        batch_nodes_coord = np.random.random([1, num_nodes, 2])
        locations = batch_nodes_coord[0]  # Take the single instance from batch
        
        # Create filename
        seed_value = seed_start + batch_idx
        filename = os.path.join(output_dir, f"instance_{num_nodes}_{seed_value}.tsp")
        
        # Write TSP file
        comment = f"Generated TSP instance N={num_nodes}, seed={seed_value}"
        write_tsp_file(filename, locations, comment)
        
        instances_generated += 1
        if instances_generated % 10 == 0:
            print(f"Generated {instances_generated} instances...")
    
    return instances_generated

def main():
    parser = argparse.ArgumentParser(description='Generate TSP instances')
    parser.add_argument('--min_nodes', type=int, default=20,
                        help='Minimum number of nodes (default: 20)')
    parser.add_argument('--max_nodes', type=int, default=100,
                        help='Maximum number of nodes (default: 100)')
    parser.add_argument('--num_instances', type=int, default=50,
                        help='Number of instances to generate (default: 100)')
    parser.add_argument('--output', type=str, default='training_instances',
                        help='Output directory (default: training_instances)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (default: None)')
    parser.add_argument('--seed_start', type=int, default=1,
                        help='Starting seed value for instance naming (default: 1)')
    
    # Alternative mode: generate fixed sizes
    parser.add_argument('--fixed_sizes', type=int, nargs='+', 
                        help='Generate fixed sizes instead of random (e.g., --fixed_sizes 50 100 200)')
    parser.add_argument('--instances_per_size', type=int, default=10,
                        help='Instances per fixed size (default: 10)')
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"Using random seed: {args.seed}")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        print(f"Created output directory: {args.output}")
    
    print(f"Generating TSP instances in {args.output} directory...")
    
    total_instances = 0
    
    if args.fixed_sizes:
        # Generate fixed sizes mode
        print(f"Fixed sizes mode: generating {args.instances_per_size} instances for each size {args.fixed_sizes}")
        
        for size in args.fixed_sizes:
            print(f"Generating {args.instances_per_size} instances with {size} nodes...")
            for i in range(args.instances_per_size):
                # Generate coordinates uniformly from unit square
                locations = np.random.random([size, 2])
                
                # Create filename
                seed_value = args.seed_start + total_instances
                filename = os.path.join(args.output, f"instance_{size}_{seed_value}.tsp")
                
                # Write TSP file
                comment = f"Generated TSP instance N={size}, seed={seed_value}"
                write_tsp_file(filename, locations, comment)
                
                total_instances += 1
            
            print(f"Done generating instances for size {size}")
    
    else:
        # Generate batch mode with random sizes
        print(f"Batch mode: generating {args.num_instances} instances with random sizes between {args.min_nodes}-{args.max_nodes} nodes")
        total_instances = generate_batch_instances(
            args.num_instances, 
            args.min_nodes, 
            args.max_nodes, 
            args.output, 
            args.seed_start
        )
    
    print(f"Successfully generated {total_instances} TSP instances in {args.output} directory")
    
    # Print summary
    if args.fixed_sizes:
        print(f"Generated {args.instances_per_size} instances each for sizes: {args.fixed_sizes}")
    else:
        print(f"Generated instances with random sizes between {args.min_nodes} and {args.max_nodes} nodes")

if __name__ == "__main__":
    main()