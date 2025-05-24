# © 2021, University of Bern, Chair of Quantitative Methods, Mario Gnägi, Philipp Baumann

# import standard packages
import time
import warnings
import numpy as np

from sklearn.neighbors import KDTree
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist, squareform

# only for kmeans++
from sklearn.cluster import kmeans_plusplus
from sklearn.utils.extmath import row_norms
from sklearn.utils import check_random_state

# import package gurobi
try:
    import gurobipy as gurobi
except ImportError:
    warnings.warn('Gurobi is not available. Please set the argument '
                  'no_solver=True when calling gb21_mh(...).')


def gb21_mh(X, Q, q, p, t_total,
            n_start, g_initial, init, n_target, l, t_local,
            mip_gap_global=0.01, mip_gap_local=0.01,
            np_seed=1, gurobi_seed=1,
            no_local=False, no_solver=False, threads=1):

    # initialize timeStart
    timeStart = time.time()

    # set numpy random seed
    np.random.seed(np_seed)

    # setup nodes collection
    nodes = np.empty(X.shape[0], dtype=NodeClass)
    for idx in np.arange(X.shape[0]):
        nodes[idx] = NodeClass(idx, X[idx], Q[idx], q[idx])

    # setup inst
    inst = InstanceClass(X.shape[0], p, X.shape[1],
                         np.arange(X.shape[0]), nodes, X)

    # initialize help variables
    initial_ofv = float('inf')
    start_counter = 0
    feas_solution_found = False

    while start_counter < n_start and time.time() - timeStart < t_total:

        # incease start_counter by 1
        start_counter += 1

        # initialize g
        g = min(g_initial, inst.p)

        # get initial medians with kmeans++
        curr_solution = SolutionClass(inst)
        curr_solution.medians = kmeans_pp(inst)
        for median in curr_solution.medians:
            curr_solution.assignments[median] = median

        # initializ curr_ofv
        curr_ofv = float('inf')

        # initialize help variables
        iteration = 0
        feasible_assignment_found = False

        # set init="capacity-based" and no_local=True
        if no_solver:
            init = "capacity-based"
            no_local = True

        # initialize heuristic_assignment (global optimization phase)
        if init == "capacity-based":
            heuristic_assignment = True
        else:
            heuristic_assignment = False

        # start global optimization phase
        while time.time() - timeStart < t_total:

            # increase iteration by 1
            iteration += 1

            # initialize new solution
            new_solution = SolutionClass(inst)
            new_solution.medians = np.array(curr_solution.medians)
            for median in new_solution.medians:
                new_solution.assignments[median] = median

            # capacity-based initialization method
            if heuristic_assignment:

                # setup kdtree
                tree = KDTree(inst.X[new_solution.medians], metric='euclidean')
                dist, ind = tree.query(inst.X, k=g)

                # determine order of assignment based on regret value
                regret_values = dist[:, 1] - dist[:, 0]
                regret_values[new_solution.medians] = 0
                assignment_order = np.argsort(regret_values)[::-1]

                # iteratively assing nodes to medians
                assigned_demand = dict.fromkeys(new_solution.medians, 0)
                for median in new_solution.medians:
                    assigned_demand[median] = inst.nodes[median].q
                for node in assignment_order:
                    time_limit_reached = False
                    if time.time() - timeStart < t_total:
                        if node not in new_solution.medians:
                            capacity_exceeded = True
                            for next_median in ind[node]:
                                median = new_solution.medians[next_median]
                                if assigned_demand[median] + \
                                   inst.nodes[node].q <= inst.nodes[median].Q:
                                    assigned_demand[median] += \
                                        inst.nodes[node].q
                                    new_solution.assignments[node] = median
                                    capacity_exceeded = False
                                    break
                            if capacity_exceeded:
                                break
                    else:
                        time_limit_reached = True
                        break
                if time_limit_reached:
                    break

                # if capacity_exceeded
                if capacity_exceeded:
                    if g < inst.p:
                        g = min(g*2, inst.p)
                        feasible_assignment_found = False
                    else:
                        if not feas_solution_found and \
                           start_counter == n_start:
                            start_counter += -1
                        break
                else:
                    feasible_assignment_found = True

            # assign nodes using mip
            else:

                # setup and solve mip
                model = setup_mip_assignment(g, t_total, mip_gap_global,
                                             gurobi_seed, inst, new_solution,
                                             timeStart)
                model.setParam('Threads', threads)
                model.optimize()

                # if model is infeasible
                if model.status == 3:
                    if g < inst.p:
                        g = min(g*2, inst.p)
                        feasible_assignment_found = False
                    else:
                        if not feas_solution_found and \
                           start_counter == n_start:
                            start_counter += -1
                        break
                elif model.SolCount == 0:  # if no solution has been found
                    break
                else:
                    feasible_assignment_found = True
                    for var in model.getVars():
                        if var.X > 0.5:
                            var_name = var.VarName
                            indices_str = var_name[
                                    var_name.find('[')+1:var_name.find(']')]
                            i, j = indices_str.split(',')
                            new_solution.assignments[int(i)] = int(j)

            # if feasible assignment found
            if feasible_assignment_found:

                # feasible solution found
                feas_solution_found = True

                # recalculate medians
                medians_changed = np.zeros(inst.p, dtype=bool)

                for k in np.arange(inst.p):
                    # identify nodes assigned to median k
                    nodes_in = np.where(
                            new_solution.assignments ==
                            curr_solution.medians[k])[0]

                    # exact median-update step
                    if inst.n/inst.p <= 10000:

                        # calculate distances and argsort
                        dist_in = squareform(pdist(inst.X[nodes_in]))
                        dist_sum = dist_in.sum(axis=0)
                        dist_argsort = np.argsort(dist_sum)

                    # approximate median-update step
                    else:

                        # calculate center of gravity
                        mean_pos = inst.X[nodes_in].sum(
                                axis=0)/nodes_in.shape[0]
                        # setup kdtree
                        tree = KDTree(inst.X[nodes_in], metric='euclidean')
                        ind = tree.query(
                                mean_pos.reshape(1, -1), k=nodes_in.shape[0],
                                return_distance=False)
                        dist_argsort = ind[0]

                    # calculate total demand assigned to median k
                    demand_in = sum([node.q for node in inst.nodes[nodes_in]])

                    # find new median with sufficient capacity
                    counter = 0
                    while demand_in > inst.nodes[
                            nodes_in[dist_argsort[counter]]].Q:
                        counter += 1
                    median = nodes_in[dist_argsort[counter]]
                    if median != curr_solution.medians[k]:
                        medians_changed[k] = True
                        new_solution.medians[k] = median

                # update indices of assignments to new medians
                for k in np.arange(inst.p):
                    if medians_changed[k]:
                        nodes_in = np.where(
                                new_solution.assignments ==
                                curr_solution.medians[k])[0]
                        new_solution.assignments[nodes_in] = \
                            new_solution.medians[k]

                # if improvement has been found
                new_ofv = get_ofv(inst, new_solution)
                if new_ofv + 0.1 < curr_ofv:
                    curr_ofv = new_ofv
                    curr_solution.assignments = \
                        np.array(new_solution.assignments)
                    curr_solution.medians = np.array(new_solution.medians)

                    # reset number of closest medians for assignment
                    g = min(g_initial, inst.p)
                else:
                    if heuristic_assignment:
                        # reset number of closest medians for assignment
                        g = min(g_initial, inst.p)
                        heuristic_assignment = False

                        # return solution if no_solver=True
                        if no_solver:
                            break
                    else:
                        break
       
        # store best solution
        if curr_ofv < initial_ofv:
            initial_ofv = curr_ofv
            initial_solution = SolutionClass(inst)
            initial_solution.medians = np.array(curr_solution.medians)
            initial_solution.assignments = np.array(curr_solution.assignments)

         # end if t_total is exceeded, return the solution found in the last round
        if time.time() - timeStart > t_total:
            # log to console
            print('{:*^60}'.format(' Global optimization phase '))
            print('Final objective: ' + '{: .4f}'.format(initial_ofv))
            print('Running time (total): ' +
                '{:.2f}s'.format(t_total))
            print('{:*^60}'.format(''))
            return initial_solution.medians, initial_solution.assignments, t_total, initial_ofv
        

    # log to console
    print('{:*^60}'.format(' Global optimization phase '))
    print('Final objective: ' + '{: .4f}'.format(initial_ofv))
    print('Running time (total): ' +
          '{:.2f}s'.format(time.time() - timeStart))
    print('{:*^60}'.format(''))

    # end if t_total is exceeded
    if time.time() - timeStart > t_total or no_local == True:

        return initial_solution.medians, initial_solution.assignments, time.time() - timeStart, initial_ofv

    # initialize best_solution
    best_ofv = initial_ofv
    best_solution = SolutionClass(inst)
    best_solution.medians = np.array(initial_solution.medians)
    best_solution.assignments = np.array(initial_solution.assignments)

    # initialize number of free medians
    w = min(max(int(np.ceil(n_target*inst.p/inst.n)), 2), inst.p)

    # initialize help variables
    iteration = 0
    full_model_flag = False
    tabu_list = np.array([], dtype=int)

    # start local optimization phase
    while time.time() - timeStart < t_total:

        # increase iteration by 1
        iteration += 1

        # select subset of medians
        subset_medians = get_subset_of_medians(inst, best_solution,
                                               tabu_list, w)
        subset_medians_pos = np.where(np.isin(
                best_solution.medians, subset_medians))[0]
        subset_nodes = np.array([node.idx for node in inst.nodes
                                 if best_solution.assignments[node.idx]
                                 in subset_medians])

        # setup and solve
        model = setup_mip_improvement(inst, best_solution, subset_medians,
                                      subset_nodes, timeStart, t_total,
                                      t_local, l, mip_gap_local, gurobi_seed)
        model.setParam('Threads', threads)
        model.optimize()

        # if full model has been solved (break after evaluation of solution)
        if w == inst.p:
            full_model_flag = True

        # if improvement has been found
        if model.objVal + 0.1 < get_ofv(
                inst, best_solution, subset_medians):

            median_counter = 0
            for var in model.getVars():
                if var.X > 0.5:
                    var_name = var.VarName
                    indices_str = \
                        var_name[var_name.find('[')+1:var_name.find(']')]
                    i, j = indices_str.split(',')
                    if i == j:
                        median_pos = subset_medians_pos[median_counter]
                        best_solution.medians[median_pos] = int(j)
                        median_counter += 1
                    best_solution.assignments[int(i)] = int(j)
            best_ofv = get_ofv(inst, best_solution)

            # update tabu_list
            tabu_list = np.setdiff1d(tabu_list, subset_medians)

        # if no improvement has been found
        else:
            # update tabu_list
            tabu_list = np.union1d(tabu_list, subset_medians)
            if np.setdiff1d(best_solution.medians, tabu_list).shape[0] == 0:
                tabu_list = np.array([], dtype=int)
                w = min(inst.p, w*2)

                # end if t_total is exceeded, return the solution found in the last round
        if time.time() - timeStart > t_total:
            # log to console
            print('{:*^60}'.format(' Local optimization phase '))
            print('Final objective: ' + '{:.4f}'.format(best_ofv))
            print('Running time (total): ' + '{:.2f}s'.format(time.time() - timeStart))
            print('{:*^60}'.format(''))

            return best_solution.medians, best_solution.assignments, t_total, best_ofv

        # break if full model has been solved
        if full_model_flag:
            break

    # log to console
    print('{:*^60}'.format(' Local optimization phase '))
    print('Final objective: ' + '{:.4f}'.format(best_ofv))
    print('Running time (total): ' + '{:.2f}s'.format(time.time() - timeStart))
    print('{:*^60}'.format(''))

    return best_solution.medians, best_solution.assignments, time.time() - timeStart, best_ofv


def kmeans_pp(inst):

    random_state = check_random_state(None)
    x_squared_norms = row_norms(inst.X, squared=True)
    centers, indices = kmeans_plusplus(inst.X, inst.p,
                                       random_state=random_state,
                                       x_squared_norms=x_squared_norms)

    return indices


def setup_mip_assignment(g, t_total, mip_gap_global, gurobi_seed,
                         inst, solution, timeStart):

    # setup kdtree
    tree = KDTree(inst.X[solution.medians], metric='euclidean')
    dist, ind = tree.query(inst.X, k=g)

    # transform dist and ind to dicts (incl. keys)
    ind = dict(zip(inst.I, solution.medians[ind]))
    dist = dict(zip(inst.I, dist))
    for node in inst.I:
        dist[node] = dict(zip(ind[node], dist[node]))

    # setup sets
    I = np.setdiff1d(inst.I, solution.medians)
    J = solution.medians
    J_ = dict.fromkeys(I, np.ndarray)
    for node in I:
        J_[node] = ind[node]
    I_ = dict.fromkeys(J, np.ndarray)
    for median in J:
        I_[median] = I[np.any(np.array([*J_.values()]) == median, axis=1)]
        I_[median] = np.setdiff1d(I_[median], J)

    # initialize model
    model = gurobi.Model('mip_assignment')

    # setup params
    model.params.LogToConsole = 0
    model.params.MIPGap = mip_gap_global
    model.params.TimeLimit = max(0, t_total - (time.time() - timeStart))
    model.params.Seed = gurobi_seed

    # initialize variables:
    x = model.addVars(
        [(i, j) for i in I for j in J_[i]],
        vtype=gurobi.GRB.BINARY, name='x')

    # define objective function
    expr = gurobi.LinExpr()
    coeffs = [dist[i][j] for i in I for j in J_[i]]
    expr.addTerms(coeffs, x.values())
    model.setObjective(expr, gurobi.GRB.MINIMIZE)

    # add constraints (7)
    for i in I:
        expr = gurobi.LinExpr()
        variables = [x[i, j] for j in J_[i]]
        coeffs = len(variables)*[1]
        expr.addTerms(coeffs, variables)
        rhs = 1
        model.addConstr(lhs=expr, sense=gurobi.GRB.EQUAL, rhs=rhs, name='c1')

    # add constraints (8)
    for j in J:
        expr = gurobi.LinExpr()
        variables = [x[i, j] for i in I_[j]]
        coeffs = [inst.nodes[i].q for i in I_[j]]
        expr.addTerms(coeffs, variables)
        rhs = inst.nodes[j].Q - inst.nodes[j].q
        model.addConstr(lhs=expr, sense=gurobi.GRB.LESS_EQUAL,
                        rhs=rhs, name='c2')

    return model


def setup_mip_improvement(inst, solution, subset_medians, subset_nodes,
                          timeStart, t_total, t_local, l, mip_gap_local,
                          gurobi_seed):

    # setup kdtree
    tree = KDTree(inst.X[subset_nodes], metric='euclidean')

    # find potential medians
    ind = tree.query(inst.X[subset_medians],
                     k=min(l, subset_nodes.shape[0]),
                     return_distance=False)
    ind_flat_unique = np.unique(ind.flatten())

    # setup subset of nodes and potential new medians
    I = subset_nodes
    J = subset_nodes[ind_flat_unique]

    # calculate distances
    dist = cdist(inst.X[I], inst.X[J])

    # initialize model
    model = gurobi.Model('mip_improvement')

    # set up parameter params
    model.params.LogToConsole = 0
    model.params.MIPGap = mip_gap_local
    model.params.TimeLimit = max(0,
                                 min(t_local,
                                     t_total -
                                     (time.time() - timeStart)))
    model.params.Seed = gurobi_seed

    # initialize variables:
    x = model.addVars(
        [(i, j) for i in I for j in J],
        vtype=gurobi.GRB.BINARY, name='x')

    # define objective function
    expr = gurobi.LinExpr()
    coeffs = dist.flatten()
    expr.addTerms(coeffs, x.values())
    model.setObjective(expr, gurobi.GRB.MINIMIZE)

    # add constraints (11)
    expr = gurobi.LinExpr()
    variables = [x[j, j] for j in J]
    coeffs = len(variables)*[1]
    expr.addTerms(coeffs, variables)
    rhs = subset_medians.shape[0]
    model.addConstr(lhs=expr, sense=gurobi.GRB.EQUAL, rhs=rhs, name='c1')

    # add constraints (12)
    for i in I:
        expr = gurobi.LinExpr()
        variables = [x[i, j] for j in J]
        coeffs = len(variables)*[1]
        expr.addTerms(coeffs, variables)
        rhs = 1
        model.addConstr(lhs=expr, sense=gurobi.GRB.EQUAL, rhs=rhs, name='c2')

    # add constraints (14)
    for i in I:
        for j in J:
            expr = gurobi.LinExpr()
            variables = [x[i, j], x[j, j]]
            coeffs = [1, -1]
            expr.addTerms(coeffs, variables)
            rhs = 0
            model.addConstr(lhs=expr, sense=gurobi.GRB.LESS_EQUAL,
                            rhs=rhs, name='c3')

    # add constraints (13)
    for j in J:
        expr = gurobi.LinExpr()
        variables = [x[i, j] for i in I]
        coeffs = [inst.nodes[i].q for i in I]
        expr.addTerms(coeffs, variables)
        variables = [x[j, j]]
        coeffs = [-inst.nodes[j].Q]
        expr.addTerms(coeffs, variables)
        rhs = 0
        model.addConstr(lhs=expr, sense=gurobi.GRB.LESS_EQUAL,
                        rhs=rhs, name='c3')

    # setup warm start
    for i in I:
        for j in J:
            if solution.assignments[i] == j:
                x[i, j].start = 1
            else:
                x[i, j].start = 0

    return model


def get_subset_of_medians(inst, solution, tabu_list, w):

    counter = 0
    subset_medians = np.empty(w, dtype=int)

    # find medians that are not tabu
    nontabu_medians = np.setdiff1d(solution.medians, tabu_list)

    # select one non-tabu median
    min_assigned_demand = float('inf')
    for median in nontabu_medians:
        assigned_nodes = np.where(solution.assignments == median)[0]
        assigned_demand = sum(
                [node.q for node in inst.nodes[assigned_nodes]])
        if assigned_demand < min_assigned_demand:
            min_assigned_demand = assigned_demand
            subset_medians[counter] = median

    # select remaining medians
    available_medians = np.setdiff1d(
            solution.medians, subset_medians[counter])
    # setup kdtree
    tree = KDTree(inst.X[available_medians], metric='euclidean')
    dist, ind = tree.query(
            inst.X[subset_medians[counter]].reshape(1, -1), k=w-1)
    subset_medians[counter+1:] = available_medians[ind]

    return subset_medians


def read_inst(inst_path):

    # open instance file
    f = open(inst_path, 'r')

    # read instance file line by line
    lineValues = f.readline().split()
    n = int(lineValues[0])
    p = int(lineValues[1])
    if len(lineValues) > 2:
        m = int(lineValues[2])
    else:
        m = 2

    X = np.empty((0, m), float)
    Q = np.empty(n, dtype=float)
    q = np.empty(n, dtype=float)
    for idx in np.arange(n):
        lineValues = f.readline().split()
        X = np.append(X, np.array([lineValues[:m]]).astype(float), axis=0)
        Q[idx] = float(lineValues[m])
        q[idx] = float(lineValues[m+1])

    return X, Q, q, p


def get_ofv(inst, solution, subset_medians=None):

    # initialize ofv
    ofv = 0

    # if only subset of nodes considered
    if subset_medians is None:
        for median in solution.medians:
            nodes = np.where(solution.assignments == median)[0]
            ofv += cdist(
                    inst.X[median].reshape(1, -1),
                    inst.X[nodes]).sum(axis=1)[0]

    # if all nodes considered
    else:
        for median in subset_medians:
            nodes = np.where(solution.assignments == median)[0]
            ofv += cdist(
                    inst.X[median].reshape(1, -1),
                    inst.X[nodes]).sum(axis=1)[0]

    return ofv


class NodeClass:
    def __init__(self, idx, feature_vector, Q, q):
        self.idx = idx
        self.feature_vector = feature_vector
        self.Q = Q
        self.q = q


class InstanceClass:
    def __init__(self, n, p, m, I, nodes, X):
        self.n = n
        self.p = p
        self.m = m
        self.I = I
        self.nodes = nodes
        self.X = X


class SolutionClass:
    def __init__(self, inst):
        self.assignments = np.full(inst.n, -1).astype(int)
        self.medians = np.array([], dtype=int)
