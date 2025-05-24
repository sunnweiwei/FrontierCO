DESCRIPTION = '''The Capacitated P-Median Problem is a facility location optimization problem where the objective is to select exactly  p  customers as medians (facility locations) and assign each customer to one of these medians to minimize the total cost, defined as the sum of the Euclidean distances between customers and their assigned medians. Each customer has a capacity Q_i. and demand q_i. If a customer is selected as the median, the total demand of the customers assigned to it cannot exceed its capacity Q_i. A feasible solution must respect this capacity constraint for all medians. Note that, each customer should be assigned to exactly one median, including the customers which are selected as the median.'''


def solve(**kwargs):
    """
    Solve the Capacitated P-Median Problem.

    This function receives the data for one problem instance via keyword arguments:
      - n (int): Number of customers/points.
      - p (int): Number of medians to choose.
      - customers (list of tuples): Each tuple is (customer_id, x, y, capacity, demand).
        Note: capacity is only relevant if the point is selected as a median.

    The goal is to select p medians (from the customers) and assign every customer to one
    of these medians so that the total cost is minimized. The cost for a customer is the
    Euclidean distance to its assigned median, and the
    total demand assigned to each median must not exceed its capacity.

    Returns:
      A dictionary with the following keys:
        - 'objective': (numeric) the total cost (objective value) computed by the algorithm.
        - 'medians': (list of int) exactly p customer IDs chosen as medians.
        - 'assignments': (list of int) a list of n integers, where the i-th integer is the customer
                         ID (from the chosen medians) assigned to customer i.
    """
    # Placeholder: Replace this with your actual implementation.
    # For now, we return an empty solution structure.
    # Your function must yield multiple solutions over time, not just return one solution
    # Use Python's yield keyword repeatedly to produce a stream of solutions
    # Each yielded solution should be better than the previous one
    while True:
        yield {
            "objective": 0,  # total cost (to be computed)
            "medians": [],  # list of p medians (customer IDs)
            "assignments": []  # list of n assignments (each is one of the medians)
        }


def load_data(input_file):
    """
    Load an instance of the Capacitated P-Median Problem from a text file.

    The input file structure is:
      Line 1: Two integers: <n> <p> (number of points and number of medians)
      n subsequent lines: <x_coordinate> <y_coordinate> <capacity> <demand>
        where capacity is only relevant if the point is selected as a median.

    Returns:
      A dictionary containing the keys:
         - 'n': int (number of points)
         - 'p': int (number of medians to select)
         - 'customers': list of tuples (customer_id, x, y, capacity, demand)
    """
    try:
        with open(input_file, 'r') as f:
            # Read non-empty lines and strip them
            lines = [line.strip() for line in f if line.strip() != '']
    except Exception as e:
        raise ValueError("Error reading input file: " + str(e))

    if not lines:
        raise ValueError("Input file is empty.")

    try:
        # Parse first line for n and p
        tokens = lines[0].split()
        if len(tokens) < 2:
            raise ValueError("First line must contain at least two values: n and p")
        n = int(tokens[0])
        p = int(tokens[1])
    except Exception as e:
        raise ValueError(f"Error parsing first line: {e}")

    # Read customer data
    customers = []
    if len(lines) < n + 1:
        raise ValueError(f"Expected {n} customer lines, but found fewer.")

    for i in range(1, n + 1):
        if i >= len(lines):
            raise ValueError(f"Missing customer data at line {i + 1}")

        tokens = lines[i].split()
        if len(tokens) < 4:
            raise ValueError(f"Invalid customer data at line {i + 1}, expected at least 4 values")

        try:
            x = float(tokens[0])
            y = float(tokens[1])
            capacity = float(tokens[2])
            demand = float(tokens[3])
            customer_id = i - 1
        except Exception as e:
            raise ValueError(f"Error parsing customer data on line {i + 1}: {e}")

        customers.append((customer_id, x, y, capacity, demand))

    case_data = {
        "n": n,
        "p": p,
        "customers": customers
    }

    return [case_data]


def eval_func(**kwargs):
    """
    Evaluate the solution for a single instance of the Capacitated P-Median Problem.

    This function expects the following keyword arguments:
      - n (int): Number of customers/points.
      - p (int): Number of medians to choose.
      - customers (list of tuples): Each tuple is (customer_id, x, y, capacity, demand).
      - objective (numeric): The objective value (total cost) reported by the solution.
      - medians (list of int): List of chosen medians (customer IDs), exactly p elements.
      - assignments (list of int): List of assignments for each customer (length n), where each entry is one of the chosen medians.

    The evaluation performs the following:
      1. Verifies that each assignment is to one of the selected medians.
      2. Checks that the total demand assigned to each median does not exceed its capacity.
      3. Recomputes the total cost as the sum, over all customers, of the Euclidean distance (rounded down)
         from the customer to its assigned median.

    Returns:
      The computed total cost (which can be compared with the reported objective value).
    """
    import math

    # Extract instance data
    n = kwargs.get("n")
    p = kwargs.get("p")
    customers = kwargs.get("customers")

    # Extract solution data
    reported_obj = kwargs.get("objective")
    medians = kwargs.get("medians")
    assignments = kwargs.get("assignments")

    if n is None or p is None or customers is None:
        raise ValueError("Instance data is incomplete.")
    if reported_obj is None or medians is None or assignments is None:
        raise ValueError("Solution data is incomplete.")

    # Validate medians length
    if len(medians) != p:
        raise ValueError(f"The solution must contain exactly {p} medians; found {len(medians)}.")

    # Validate assignments length
    if len(assignments) != n:
        raise ValueError(f"The solution must contain exactly {n} assignments; found {len(assignments)}.")

    # Build a dictionary for quick lookup of customer data by customer_id.
    cust_dict = {}
    for cust in customers:
        cid, x, y, capacity, demand = cust
        cust_dict[cid] = (x, y, capacity, demand)

    # Verify that each median is a valid customer.
    for m in medians:
        if m not in cust_dict:
            raise ValueError(f"Median {m} is not found in the customer data.")

    # Verify that each customer's assignment is one of the selected medians.
    for idx, a in enumerate(assignments):
        if a not in medians:
            raise ValueError(
                f"Customer {idx + 1} is assigned to {a} which is not in the list of selected medians.")

    # Check capacity constraints.
    capacity_usage = {m: 0.0 for m in medians}
    for i, a in enumerate(assignments):
        # Get customer demand
        _, _, _, _, customer_demand = customers[i]
        capacity_usage[a] += customer_demand

    for m, used in capacity_usage.items():
        # Get median capacity
        _, _, median_capacity, _ = cust_dict[m]
        if used > median_capacity + 1e-6:  # small tolerance
            raise ValueError(
                f"Capacity exceeded for median {m}: used capacity {used:.4f} exceeds allowed capacity {median_capacity:.4f}.")

    # Recompute the total cost.
    total_cost = 0
    for i, a in enumerate(assignments):
        # Get customer i data.
        try:
            cid, cx, cy, _, _ = customers[i]
        except Exception as e:
            raise ValueError(f"Error accessing data for customer {i + 1}: {e}")
        # Get the assigned median's coordinates.
        if a not in cust_dict:
            raise ValueError(f"Assigned median {a} for customer {i + 1} not found.")
        mx, my, _, _ = cust_dict[a]
        d = math.sqrt((cx - mx) ** 2 + (cy - my) ** 2)
        total_cost += math.floor(d)

    if total_cost <= 0:
        raise ValueError("Computed total cost is non-positive, which is invalid.")

    return total_cost


def get_bks(instance_file_path):
    """
    Get the best known solution (PrimalBound) for a given instance from results_summary.csv.
    
    This function:
    1. Extracts the parent directory of the instance file
    2. Modifies the directory by adding "_sol" prefix
    3. Reads the results_summary.csv file in the modified directory
    4. Extracts the PrimalBound value for the given instance
    
    Args:
        instance_file_path (str): Path to the instance file
        
    Returns:
        float: The best known solution (PrimalBound) for the instance
    """
    import os
    import csv

    # Extract the instance file name and parent directory
    instance_file_name = os.path.basename(instance_file_path)
    parent_dir = os.path.dirname(instance_file_path)

    # Create the modified directory path with "_sol" prefix
    if parent_dir:
        # If there's a parent directory, add "_sol" to it
        base_dir_name = os.path.basename(parent_dir)
        grandparent_dir = os.path.dirname(parent_dir)
        sol_dir = os.path.join(grandparent_dir, base_dir_name + "_sol")
    else:
        # If the instance file is in the current directory
        sol_dir = "instances_sol"

    # Path to the summary_results.csv file
    results_file = os.path.join(sol_dir, "summary_results.csv")

    # Read the CSV file to find the PrimalBound for the instance
    try:
        with open(results_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Check if this row corresponds to our instance
                if row['Instance'] == instance_file_name:
                    # Return the PrimalBound as a float
                    return float(row['PrimalBound'])

            # If we get here, the instance wasn't found
            raise ValueError(f"Instance {instance_file_name} not found in {results_file}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Results file not found: {results_file}")
    except Exception as e:
        raise Exception(f"Error reading results file: {e}")


def norm_score(results):
    optimal_scores = {'easy_test_instances/SJC2.txt': [33270.94], 'easy_test_instances/SJC1.txt': [17288.99], 'easy_test_instances/SJC3b.txt': [40635.9], 'easy_test_instances/SJC3a.txt': [45335.16], 'easy_test_instances/SJC4b.txt': [52458.02], 'easy_test_instances/SJC4a.txt': [61925.51], 'easy_test_instances/fnl4461_0020.txt': [1283536.73], 'easy_test_instances/fnl4461_0100.txt': [548909.01], 'easy_test_instances/fnl4461_0250.txt': [335888.87], 'easy_test_instances/fnl4461_1000.txt': [145862.38], 'easy_test_instances/fnl4461_0500.txt': [224662.49], 'easy_test_instances/p3038_1000.txt': [85854.05], 'easy_test_instances/p3038_600.txt': [122711.17], 'easy_test_instances/pr2392_150.txt': [711111.25], 'easy_test_instances/pr2392_020.txt': [2231212.51686], 'easy_test_instances/pr2392_075.txt': [1091982.77344], 'easy_test_instances/p3038_700.txt': [109677.3], 'easy_test_instances/rl1304_010.txt': [2146252.17038], 'easy_test_instances/p3038_800.txt': [100064.94], 'easy_test_instances/p3038_900.txt': [92310.09], 'easy_test_instances/pr2392_300.txt': [458145.29], 'easy_test_instances/rl1304_100.txt': [498090.74], 'easy_test_instances/rl1304_050.txt': [802283.41], 'easy_test_instances/rl1304_200.txt': [276977.6], 'easy_test_instances/u724_010.txt': [181782.96], 'easy_test_instances/u724_030.txt': [95034.01], 'easy_test_instances/u724_075.txt': [54735.05], 'easy_test_instances/rl1304_300.txt': [191224.85], 'easy_test_instances/u724_125.txt': [38976.76], 'easy_test_instances/u724_200.txt': [28079.97], 'easy_test_instances/pr2392_500.txt': [316042.97]}
    optimal_scores = optimal_scores | {'hard_test_instances/FNA52057_2000.txt': [410025.26], 'hard_test_instances/FNA52057_1000.txt': [629746.16], 'hard_test_instances/FNA52057_100.txt': [2099669.49], 'hard_test_instances/LRA498378_100.txt': [103567233.39], 'hard_test_instances/LRA498378_1000.txt': [30293246.998271], 'hard_test_instances/SRA104814_100.txt': [4768046.624651], 'hard_test_instances/SRA104814_1000.txt': [1482994.81], 'hard_test_instances/LRA498378_2000.txt': [21136297.537245], 'hard_test_instances/SRA104814_2000.txt': [1003547.20934], 'hard_test_instances/XMC10150_100.txt': [181472.17], 'hard_test_instances/XMC10150_1000.txt': [46549.377303], 'hard_test_instances/XMC10150_2000.txt': [27546.961844]}
    optimal_scores = optimal_scores | {'valid_instances/p_median_n500_p20_1.txt': [3960.394336], 'valid_instances/p_median_n500_p10_2.txt': [5813.675331], 'valid_instances/p_median_n500_p10_1.txt': [5737.960126], 'valid_instances/p_median_n500_p10_3.txt': [6016.16501], 'valid_instances/p_median_n500_p10_5.txt': [5782.510617], 'valid_instances/p_median_n500_p20_4.txt': [3981.271501], 'valid_instances/p_median_n500_p10_4.txt': [5922.286345], 'valid_instances/p_median_n500_p20_5.txt': [3880.373158], 'valid_instances/p_median_n500_p50_3.txt': [2260.498576], 'valid_instances/p_median_n500_p5_3.txt': [8636.247659], 'valid_instances/p_median_n500_p5_1.txt': [8756.853525], 'valid_instances/p_median_n500_p5_2.txt': [8684.292988], 'valid_instances/p_median_n500_p5_4.txt': [8257.644537], 'valid_instances/p_median_n500_p5_5.txt': [8571.71517], 'valid_instances/p_median_n500_p20_2.txt': [3995.152147], 'valid_instances/p_median_n500_p20_3.txt': [4010.296989], 'valid_instances/p_median_n500_p50_5.txt': [2257.273104], 'valid_instances/p_median_n500_p50_1.txt': [2303.789728], 'valid_instances/p_median_n500_p50_2.txt': [2316.964534], 'valid_instances/p_median_n500_p50_4.txt': [2330.859137]}

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