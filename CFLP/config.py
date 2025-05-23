DESCRIPTION = '''The Capacitated Facility Location Problem aims to determine which facilities to open and how to allocate portions of customer demands among these facilities in order to minimize total costs. Given a set of potential facility locations, each with a fixed opening cost and capacity limit, and a set of customers with individual demands and associated assignment costs to each facility, the objective is to decide which facilities to open and how to distribute each customer's demand among these open facilities. The allocation must satisfy the constraint that the sum of portions assigned to each customer equals their total demand, and that the total demand allocated to any facility does not exceed its capacity. The optimization seeks to minimize the sum of fixed facility opening costs and the total assignment costs. However, if any solution violates these constraints (i.e., a customer’s demand is not fully satisfied or a warehouse’s capacity is exceeded), then an infitiely large cost is given.'''


def solve(**kwargs):
    """
    Solves the Capacitated Facility Location Problem.

    Input kwargs:
  - n (int): Number of facilities
  - m (int): Number of customers
  - capacities (list): A list of capacities for each facility
  - fixed_cost (list): A list of fixed costs for each facility
  - demands (list): A list of demands for each customer
  - trans_costs (list of list): A 2D list of transportation costs, where trans_costs[i][j] represents 
                               the cost of allocating the entire demand of customer j to facility i

    Note: The input structure should match the output of load_data function.

    Evaluation Metric:
      The objective is to minimize the total cost, computed as:
         (Sum of fixed costs for all open facilities)
       + (Sum of transportation costs for customer demand allocated from facilities to customers)
      For each customer, the sum of allocations from all facilities must equal the customer's demand.
      For each facility, the total allocated demand across all customers must not exceed its capacity.
      If a solution violates any of these constraints, the solution is considered infeasible and no score is provided.

    Returns:
      A dictionary with the following keys:
         'total_cost': (float) The computed objective value (cost) if the solution is feasible;
                         otherwise, no score is provided.
         'facilities_open': (list of int) A list of n integers (0 or 1) indicating whether each facility is closed or open.
         'assignments': (list of list of float) A 2D list (m x n) where each entry represents the amount of customer i's demand supplied by facility j.
    """
    ## placeholder. You do not need to write anything here.
    return {
        "total_cost": 0.0,
        "facilities_open": [0] * kwargs["n"],
        "assignments": [[0.0] * kwargs["n"] for _ in range(kwargs["m"])]
    }

def load_data(input_path):
    """Read Capacitated Facility Location Problem instance from a file.
    
    New format:
    n, m (n=facilities, m=customers)
    b1, f1 (capacity and fixed cost of facility 1)
    b2, f2
    ...
    bn, fn
    d1, d2, d3, ..., dm (customer demands)
    c11, c12, c13, ..., c1m (allocation costs for facility 1 to all customers)
    c21, c22, c23, ..., c2m
    ...
    cn1, cn2, cn3, ..., cnm
    """
    # Read all numbers from the file
    with open(filename, 'r') as f:
        content = f.read()
        # Extract all numbers, ignoring whitespace and empty lines
        all_numbers = [num for num in content.split() if num.strip()]
        
    pos = 0  # Position in the numbers list
    
    # Parse dimensions: n (facilities), m (customers)
    n = int(all_numbers[pos])
    pos += 1
    m = int(all_numbers[pos])
    pos += 1
    
    # Parse facility data: capacity, fixed cost
    capacities = []
    fixed_costs = []
    for _ in range(n):
        if pos + 1 < len(all_numbers):
            capacities.append(float(all_numbers[pos]))
            pos += 1
            fixed_costs.append(float(all_numbers[pos]))
            pos += 1
    
    # Parse customer demands
    demands = []
    for _ in range(m):
        if pos < len(all_numbers):
            demands.append(float(all_numbers[pos]))
            pos += 1
    
    # Parse transportation costs
    trans_costs = []
    for _ in range(n):
        facility_costs = []
        for _ in range(m):
            if pos < len(all_numbers):
                facility_costs.append(float(all_numbers[pos]))
                pos += 1
        trans_costs.append(facility_costs)
    
    # Verify that we have the expected amount of data
    expected_numbers = 2 + 2*n + m + n*m
    if len(all_numbers) < expected_numbers:
        print(f"Warning: File might be incomplete. Expected {expected_numbers} numbers, found {len(all_numbers)}.")

    case = {"n": m, "m": m, "capacities": capacities "fixed_cost": fixed_cost, "demands": demands, 'trans_costs', trans_costs}

    return case


def eval_func(n, m, capacities, fixed_cost, demands, trans_costs, facilities_open, assignments, **kwargs):
    """
    Evaluates the solution for the Capacitated facility Location Problem with Splittable Customer Demand,
    using a weighted average cost for each customer.

    For each customer:
      - The sum of allocations across facilities must equal the customer's demand.
      - The assignment cost is computed as the weighted average of the per-unit costs,
        i.e., for each facility i, the fraction of demand allocated from i multiplied by its cost.
      - No positive allocation is allowed for a facility that is closed.

    Additionally, for each facility:
      - The total allocated demand must not exceed its capacity.

    The total cost is computed as:
         (Sum of fixed costs for all open facilities)
       + (Sum over customers of the weighted average assignment cost)

    Input Parameters:
      - n: Number of facilities (int)
      - m: Number of customers (int)
      - capacities: List of capacities for each facility (list of float)
      - fixed_cost: List of fixed costs for each facility (list of float)
      - demands: List of demands for each customer (list of float)
      - trans_costs: List of lists representing transportation costs from facilities to customers
      - facilities_open: List of n integers (0 or 1) indicating whether each facility is closed or open
      - assignments: List of m lists (each of length n) where assignments[j][i] represents the amount of
                     customer j's demand allocated to facility i
      - kwargs: Other parameters (not used here)

    Returns:
      A floating-point number representing the total cost if the solution is feasible.

    Raises:
      Exception: If any of the following conditions are violated:
          - The sum of allocations for any customer does not equal its demand.
          - Any positive allocation is made to a closed facility.
          - Any facility's total allocated demand exceeds its capacity.
    """
    computed_total_cost = 0.0

    # Add fixed costs for open facilities.
    for i in range(n):
        if facilities_open[i] == 1:
            computed_total_cost += fixed_cost[i]

    # Evaluate assignment cost for each customer as a weighted average.
    for j in range(m):
        customer_demand = demands[j]
        allocated_amount = sum(assignments[j])
        if abs(allocated_amount - customer_demand) > 1e-6:
            raise Exception(
                f"Customer {j} demand violation: total assigned amount {allocated_amount} does not equal demand {customer_demand}."
            )
        weighted_cost = 0.0
        for i in range(n):
            allocation = assignments[j][i]
            if allocation < 0:
                raise Exception(
                    f"Customer {j} has a negative allocation {allocation} for facility {i}."
                )
            if allocation > 0 and facilities_open[i] != 1:
                raise Exception(
                    f"Customer {j} has allocation {allocation} for facility {i}, which is closed."
                )
            # Compute fraction of the customer's demand supplied from facility i.
            fraction = allocation / customer_demand if customer_demand > 0 else 0.0
            weighted_cost += fraction * trans_costs[i][j]
        # Add the weighted cost (applied once per customer).
        computed_total_cost += weighted_cost

    # Compute total demand allocated to each facility and check capacity constraints.
    assigned_demand = [0.0] * n
    for i in range(n):
        for j in range(m):
            assigned_demand[i] += assignments[j][i]
    for i in range(n):
        if assigned_demand[i] > capacities[i] + 1e-6:
            excess = assigned_demand[i] - capacities[i]
            raise Exception(
                f"Facility {i} exceeds its capacity by {excess} units."
            )

    return computed_total_cost