DESCRIPTION = '''The Flexible Job Shop Scheduling Problem (FJSP) aims to assign operations of jobs to compatible machines and determine their processing sequence to minimize the makespan (total completion time). Given a set of jobs, each consisting of a sequence of operations, and a set of machines, where each operation can be processed on one or more machines with potentially different processing times, the objective is to:
1. Assign each operation to exactly one compatible machine
2. Determine the processing sequence of operations on each machine
3. Minimize the makespan (completion time of the last operation)

The problem has the following constraints:
- Each operation must be processed on exactly one machine from its set of compatible machines
- Operations of the same job must be processed in their predefined order (precedence constraints)
- Each machine can process only one operation at a time
- No preemption is allowed (once an operation starts, it must finish without interruption)
- All jobs are available at time zero'''



def solve(num_jobs, num_machines, jobs, **kwargs):
    """
    Solves the Flexible Job Shop Scheduling Problem using constraint programming.
    
    Args:
        num_jobs (int): Number of jobs
        num_machines (int): Number of machines
        jobs (list): A list of jobs, where each job is a list of operations
                    Each operation is represented as a list of machine-time pairs
        
    Returns:
        dict: A dictionary containing the makespan, machine assignments, and start times
    """
    try:
        # Try to import the docplex library
        from docplex.cp.model import CpoModel
    except ImportError:
        raise ImportError("This implementation requires the 'docplex' package. Please install it with 'pip install docplex'.")
    
    # Create a model instance
    mdl = CpoModel()
    
    # Convert input to the required format for our model
    class Instance:
        def __init__(self, num_jobs, num_machines, jobs):
            self.n = num_jobs  # Number of jobs
            self.g = num_machines + 1  # Number of machines (add 1 because machine indices start from 1)
            
            # Create list to store number of operations for each job
            self.o = [len(job) for job in jobs]
            
            # Create processing time matrix p[j][k][i] = processing time of job j, operation k on machine i
            self.p = []
            for j in range(self.n):
                job_times = []
                for k in range(self.o[j]):
                    # Initialize processing times for all machines to 0
                    machine_times = [0] * self.g
                    
                    # Fill in valid processing times
                    for machine, time in jobs[j][k]:
                        machine_times[machine] = time
                    
                    job_times.append(machine_times)
                self.p.append(job_times)
    
    # Create instance
    instance = Instance(num_jobs, num_machines, jobs)
    
    # Initialize the interval variables for each operation on each possible machine
    tasks = []
    for j in range(instance.n):
        job_tasks = []
        for k in range(instance.o[j]):
            # Create interval variable for each possible machine
            operation_tasks = []
            for i in range(instance.g):
                # Only create variables for compatible machines (where processing time > 0)
                if instance.p[j][k][i] > 0:
                    task = mdl.interval_var(
                        name=f"A_{j}_{k}_{i}",
                        optional=True,
                        size=instance.p[j][k][i]
                    )
                    operation_tasks.append((i, task))  # Store machine index with the task
                
            job_tasks.append(operation_tasks)
        tasks.append(job_tasks)
    
    # Create interval variables for each job operation (regardless of machine)
    job_tasks = []
    for j in range(instance.n):
        job_op_tasks = []
        for k in range(instance.o[j]):
            task = mdl.interval_var(name=f"T_{j}_{k}")
            job_op_tasks.append(task)
        job_tasks.append(job_op_tasks)
    
    # Add alternative constraints - each operation must be assigned to exactly one machine
    for j in range(instance.n):
        for k in range(instance.o[j]):
            # Get all machine tasks for this operation
            machine_tasks = [task for _, task in tasks[j][k]]
            if machine_tasks:  # Only add constraint if there are machine options
                mdl.add(mdl.alternative(job_tasks[j][k], machine_tasks))
    
    # Add no-overlap constraints for each machine
    for i in range(1, instance.g):  # Start from 1 because machine indices start from 1
        # Get all tasks that can be processed on machine i
        machine_tasks = []
        for j in range(instance.n):
            for k in range(instance.o[j]):
                for machine_idx, task in tasks[j][k]:
                    if machine_idx == i:
                        machine_tasks.append(task)
        
        # Add no-overlap constraint if there are tasks for this machine
        if machine_tasks:
            mdl.add(mdl.no_overlap(machine_tasks))
    
    # Add precedence constraints for operations within the same job
    for j in range(instance.n):
        for k in range(1, instance.o[j]):
            mdl.add(mdl.end_before_start(job_tasks[j][k-1], job_tasks[j][k]))
    
    # Set objective: minimize makespan
    makespan = mdl.max([mdl.end_of(job_tasks[j][-1]) for j in range(instance.n)])
    mdl.add(mdl.minimize(makespan))
    
    # Solve the model
    solution = mdl.solve(TimeLimit=60)  # 60 seconds time limit
    
    if not solution:
        raise Exception("No solution found within the time limit")
    
    # Extract machine assignments and start times
    machine_assignments = []
    start_times = []
    
    # Helper to find which machine was selected for an operation
    def find_selected_machine(j, k):
        for machine_idx, task in tasks[j][k]:
            if solution.get_var_solution(task).is_present():
                return machine_idx
        return None
    
    # Build the solution in the required format (operations are indexed globally)
    for j in range(instance.n):
        for k in range(instance.o[j]):
            # Find which machine was assigned to this operation
            machine = find_selected_machine(j, k)
            machine_assignments.append(machine)
            
            # Get the start time
            start = solution.get_var_solution(job_tasks[j][k]).start
            start_times.append(start)
    
    # Calculate the actual makespan
    actual_makespan = solution.get_objective_values()[0]
    
    return {
        "makespan": actual_makespan,
        "machine_assignments": machine_assignments,
        "start_times": start_times
    }

def load_data(filename):
    """Read Flexible Job Shop Scheduling Problem instance from a file.
    
    Format:
    <number of jobs> <number of machines>
    <number of operations for job 1> <number of machines for op 1> <machine 1> <time 1> <machine 2> <time 2> ... <number of machines for op 2> <machine 1> <time 1> ...
    <number of operations for job 2> ...
    ...
    
    Example:
    3 5
    2 2 1 3 2 5 3 1 3 2 4 6
    3 1 4 4 2 3 1 5 2 2 4 5 3
    2 2 1 5 3 4 3 2 3 5 2
    
    This example has 3 jobs and 5 machines.
    Job 1 has 2 operations:
      - Operation 1 can be processed on 2 machines: machine 1 (time 3) or machine 2 (time 5)
      - Operation 2 can be processed on 3 machines: machine 1 (time 3), machine 2 (time 4), or machine 4 (time 6)
    And so on for jobs 2 and 3.
    """
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    # Parse first line: number of jobs and machines
    parts = lines[0].split()
    num_jobs = int(parts[0])
    num_machines = int(parts[1])
    
    # Parse job information
    jobs = []
    for i in range(1, num_jobs + 1):
        if i < len(lines):
            job_data = list(map(int, lines[i].split()))
            job_operations = []
            
            # Parse operations for this job
            idx = 1  # Skip the first number (number of operations)
            num_operations = job_data[0]
            
            for _ in range(num_operations):
                if idx < len(job_data):
                    num_machines_for_op = job_data[idx]
                    idx += 1
                    
                    # Parse machine-time pairs for this operation
                    machine_time_pairs = []
                    for _ in range(num_machines_for_op):
                        if idx + 1 < len(job_data):
                            machine = job_data[idx]
                            time = job_data[idx + 1]
                            machine_time_pairs.append((machine, time))
                            idx += 2
                    
                    job_operations.append(machine_time_pairs)
            
            jobs.append(job_operations)
    
    # Validate that we have the expected amount of data
    if len(jobs) != num_jobs:
        print(f"Warning: Expected {num_jobs} jobs, found {len(jobs)}.")
    
    case = {
        "num_jobs": num_jobs,
        "num_machines": num_machines,
        "jobs": jobs
    }
    
    return case


def eval_func(num_jobs, num_machines, jobs, machine_assignments, start_times, **kwargs):
    """
    Evaluates the solution for the Flexible Job Shop Scheduling Problem.
    
    Input Parameters:
      - num_jobs (int): Number of jobs
      - num_machines (int): Number of machines
      - jobs (list): A list of jobs, where each job is a list of operations
      - machine_assignments (list): A list of machine assignments for each operation
      - start_times (list): A list of start times for each operation
      - kwargs: Other parameters (not used here)
    
    Returns:
      A floating-point number representing the makespan if the solution is feasible.
    
    Raises:
      Exception: If any constraint is violated.
    """
    # Flatten job operations for indexing
    flat_operations = []
    for job in jobs:
        for operation in job:
            flat_operations.append(operation)
    
    # Validate machine assignments
    for i, (operation, assigned_machine) in enumerate(zip(flat_operations, machine_assignments)):
        # Check if assigned machine is compatible with operation
        compatible_machines = [machine for machine, _ in operation]
        if assigned_machine not in compatible_machines:
            raise Exception(f"Operation {i} assigned to incompatible machine {assigned_machine}")
    
    # Track job precedence constraints
    job_op_end_times = {}  # (job_idx, op_idx_within_job) -> end_time
    op_idx = 0
    
    # Calculate end times and check precedence constraints
    for job_idx, job in enumerate(jobs):
        for op_idx_within_job, operation in enumerate(job):
            current_op_idx = op_idx
            
            # Get assigned machine and processing time
            assigned_machine = machine_assignments[current_op_idx]
            processing_time = next(time for machine, time in operation if machine == assigned_machine)
            
            start_time = start_times[current_op_idx]
            end_time = start_time + processing_time
            
            # Check job precedence constraint
            if op_idx_within_job > 0:
                prev_end_time = job_op_end_times.get((job_idx, op_idx_within_job - 1), 0)
                if start_time < prev_end_time:
                    raise Exception(f"Operation {current_op_idx} starts at {start_time}, " 
                                   f"before previous operation in job {job_idx} ends at {prev_end_time}")
            
            job_op_end_times[(job_idx, op_idx_within_job)] = end_time
            op_idx += 1
    
    # Check machine capacity constraints (no overlap on same machine)
    machine_schedules = {}  # machine -> list of (start_time, end_time) tuples
    op_idx = 0
    
    for job in jobs:
        for operation in job:
            assigned_machine = machine_assignments[op_idx]
            processing_time = next(time for machine, time in operation if machine == assigned_machine)
            
            start_time = start_times[op_idx]
            end_time = start_time + processing_time
            
            if assigned_machine not in machine_schedules:
                machine_schedules[assigned_machine] = []
            
            # Check for overlaps on this machine
            for other_start, other_end in machine_schedules[assigned_machine]:
                if not (end_time <= other_start or start_time >= other_end):
                    raise Exception(f"Operation at time {start_time}-{end_time} overlaps with another " 
                                   f"operation on machine {assigned_machine} at {other_start}-{other_end}")
            
            machine_schedules[assigned_machine].append((start_time, end_time))
            op_idx += 1
    
    # Calculate makespan
    makespan = max(end_time for machine_times in machine_schedules.values() 
                   for _, end_time in machine_times)
    
    return makespan


# Example usage:
def test_solver():
    """Test the solver with a small example."""
    # Example data (3 jobs, 5 machines)
    test_case = load_data('/usr1/data/shengyuf/FJSP/easy_test_instances/Behnke9.fjs')
    # Call the solver
    solution = solve(**test_case)
    
    # Print the solution
    print(f"Makespan: {solution['makespan']}")
    print(f"Machine assignments: {solution['machine_assignments']}")
    print(f"Start times: {solution['start_times']}")
    
    # Validate the solution
    try:
        makespan = eval_func(**test_case, 
                           machine_assignments=solution['machine_assignments'], 
                           start_times=solution['start_times'])
        print(f"Solution is valid with makespan: {makespan}")
    except Exception as e:
        print(f"Solution is invalid: {e}")



if __name__ == "__main__":
    # You could add code here to load and solve a specific instance
    test_solver()