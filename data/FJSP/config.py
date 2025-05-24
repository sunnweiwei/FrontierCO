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


def solve(**kwargs):
    """
    Solves the Flexible Job Shop Scheduling Problem.

    Input kwargs:
    - num_jobs (int): Number of jobs
    - num_machines (int): Number of machines
    - jobs (list): A list of jobs, where each job is a list of operations
                  Each operation is represented as a list of machine-time pairs:
                  [[(machine1, time1), (machine2, time2), ...], ...]
                  where machine_i is the index of a compatible machine and time_i is the processing time

    Note: The input structure should match the output of load_data function.
    Note: Items are always 1-indexed

    Evaluation Metric:
      The objective is to minimize the makespan (completion time of the last operation).
      The solution must satisfy all constraints:
      - Each operation is assigned to exactly one compatible machine
      - Operations of the same job are processed in their predefined order
      - Each machine processes only one operation at a time
      - No preemption is allowed

    Returns:
      A dictionary with the following keys:
         'makespan': (float) The completion time of the last operation
         'machine_assignments': (list) A list where each element i represents the machine assigned to operation i
                               (operations are indexed globally, in order of job and then operation index)
         'start_times': (list) A list where each element i represents the start time of operation i
    """
    ## placeholder. You do not need to write anything here.
    # Your function must yield multiple solutions over time, not just return one solution
    # Use Python's yield keyword repeatedly to produce a stream of solutions
    # Each yielded solution should be better than the previous one
    while True:
        yield {
            "makespan": 0.0,
            "machine_assignments": [],
            "start_times": []
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

    return [case]



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


def norm_score(results):
    optimal_scores = {'easy_test_instances/Behnke1.fjs': [90.0], 'easy_test_instances/Behnke10.fjs': [127.0], 'easy_test_instances/Behnke11.fjs': [231.0], 'easy_test_instances/Behnke12.fjs': [220.0], 'easy_test_instances/Behnke13.fjs': [231.0], 'easy_test_instances/Behnke14.fjs': [232.0], 'easy_test_instances/Behnke15.fjs': [227.0], 'easy_test_instances/Behnke16.fjs': [417.0], 'easy_test_instances/Behnke17.fjs': [406.0], 'easy_test_instances/Behnke18.fjs': [404.0], 'easy_test_instances/Behnke19.fjs': [407.0], 'easy_test_instances/Behnke2.fjs': [91.0], 'easy_test_instances/Behnke20.fjs': [404.0], 'easy_test_instances/Behnke21.fjs': [85.0], 'easy_test_instances/Behnke22.fjs': [87.0], 'easy_test_instances/Behnke23.fjs': [85.0], 'easy_test_instances/Behnke24.fjs': [87.0], 'easy_test_instances/Behnke25.fjs': [87.0], 'easy_test_instances/Behnke26.fjs': [113.0], 'easy_test_instances/Behnke27.fjs': [122.0], 'easy_test_instances/Behnke28.fjs': [114.0], 'easy_test_instances/Behnke29.fjs': [118.0], 'easy_test_instances/Behnke3.fjs': [91.0], 'easy_test_instances/Behnke30.fjs': [121.0], 'easy_test_instances/Behnke31.fjs': [226.0], 'easy_test_instances/Behnke32.fjs': [222.0], 'easy_test_instances/Behnke33.fjs': [226.0], 'easy_test_instances/Behnke34.fjs': [221.0], 'easy_test_instances/Behnke35.fjs': [214.0], 'easy_test_instances/Behnke36.fjs': [392.0], 'easy_test_instances/Behnke37.fjs': [399.0], 'easy_test_instances/Behnke38.fjs': [395.0], 'easy_test_instances/Behnke39.fjs': [393.0], 'easy_test_instances/Behnke4.fjs': [97.0], 'easy_test_instances/Behnke40.fjs': [421.0], 'easy_test_instances/Behnke41.fjs': [87.0], 'easy_test_instances/Behnke42.fjs': [87.0], 'easy_test_instances/Behnke43.fjs': [86.0], 'easy_test_instances/Behnke44.fjs': [84.0], 'easy_test_instances/Behnke45.fjs': [87.0], 'easy_test_instances/Behnke46.fjs': [115.0], 'easy_test_instances/Behnke47.fjs': [117.0], 'easy_test_instances/Behnke48.fjs': [125.0], 'easy_test_instances/Behnke49.fjs': [113.0], 'easy_test_instances/Behnke5.fjs': [91.0], 'easy_test_instances/Behnke50.fjs': [124.0], 'easy_test_instances/Behnke51.fjs': [220.0], 'easy_test_instances/Behnke52.fjs': [215.0], 'easy_test_instances/Behnke53.fjs': [213.0], 'easy_test_instances/Behnke54.fjs': [225.0], 'easy_test_instances/Behnke55.fjs': [222.0], 'easy_test_instances/Behnke56.fjs': [394.0], 'easy_test_instances/Behnke57.fjs': [393.0], 'easy_test_instances/Behnke58.fjs': [406.0], 'easy_test_instances/Behnke59.fjs': [404.0], 'easy_test_instances/Behnke6.fjs': [125.0], 'easy_test_instances/Behnke60.fjs': [402.0], 'easy_test_instances/Behnke7.fjs': [125.0], 'easy_test_instances/Behnke8.fjs': [124.0], 'easy_test_instances/Behnke9.fjs': [127.0], 'hard_test_instances/73.txt': [3723.0], 'hard_test_instances/74.txt': [3706.0], 'hard_test_instances/75.txt': [3436.0], 'hard_test_instances/76.txt': [3790.0], 'hard_test_instances/77.txt': [7406.0], 'hard_test_instances/78.txt': [7570.0], 'hard_test_instances/79.txt': [7040.0], 'hard_test_instances/80.txt': [7825.0], 'hard_test_instances/81.txt': [2276.0], 'hard_test_instances/82.txt': [2520.0], 'hard_test_instances/83.txt': [2290.0], 'hard_test_instances/84.txt': [2581.0], 'hard_test_instances/85.txt': [4901.0], 'hard_test_instances/86.txt': [5109.0], 'hard_test_instances/87.txt': [4954.0], 'hard_test_instances/88.txt': [4994.0], 'hard_test_instances/89.txt': [1810.0], 'hard_test_instances/90.txt': [1778.0], 'hard_test_instances/91.txt': [1707.0], 'hard_test_instances/92.txt': [1923.0], 'hard_test_instances/93.txt': [3553.0], 'hard_test_instances/94.txt': [3790.0], 'hard_test_instances/95.txt': [3586.0], 'hard_test_instances/96.txt': [3896.0]}

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

