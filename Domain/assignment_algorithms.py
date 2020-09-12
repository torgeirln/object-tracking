import numpy as np


def montecarlo_assignment(L, M, N):
    """ L - Cost matrix
        M - Number of associations to return
        N - Number of Monte Carlo experiments """
    assert N >= M
    n = L.shape[0] # Number of objects
    m = L.shape[1] - n # Number of measurements
    As = np.zeros((N,n,m+n))
    As_list = []
    Js = np.zeros(N)
    # Run Monte Carlo Experiments
    for i in range(N):
        A = random_assignemnt(n, m)
        new_A = True
        for j in range(i):
            if ((A - As[j,:,:]) == np.zeros((n,m+n))).all():
                new_A = False
                break
        if new_A:
            As[i,:,:] = A
            As_list.append(A)
            Js[i] = compute_cost(L, As[i,:,:])
        else:
            Js[i] = np.inf
    # Find M best associations
    best_inds = np.argsort(Js)
    return As[best_inds[:M],:,:], Js[best_inds[:M]]

def random_assignemnt(n, m):
    A = np.zeros((n,m+n))
    obj_inds = np.arange(n)
    association_inds = np.arange(m+n)
    for _ in range(n):
        rand_obj = np.random.choice(obj_inds)
        obj_inds = obj_inds[obj_inds != rand_obj]
        rand_assoc = np.random.choice(association_inds)
        if rand_assoc >= m: # Association to undetected object
            rand_assoc = m + rand_obj
        association_inds = association_inds[association_inds != rand_assoc]
        A[rand_obj, rand_assoc] = 1
    return A

def compute_cost(L, A):
    costs = L * A
    costs = costs[~np.isnan(costs)]
    return sum(costs) 
