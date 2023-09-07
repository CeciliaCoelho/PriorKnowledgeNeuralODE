import torch
import math

def dynamicAdaptive(MSE_theta, v_j, itr, MSE_thetaf, flag=0):
    """
    Adaptation of the dynamic penalty function proposed in R. B. Francisco, M. F. P. Costa, and A. M. A. Rocha, “A firefly dynamic penalty approach for solving engineering design problems” 

    Args:
    MSE_theta (float): evaluation of the objective function f in the current point theta
    v_j (list of lists): list of constraints violations at each time step
    itr (int): optimization iteration number
    MSE_thetaf (float): value of objective function f if the point is feasible till this iteration
    flag (int): choice betweeen penalty parameter update rule

    Returns:
    phi_theta (float):
    MSE_thetaf (float):
    """
    tol = 1e-4
    C = 0.5
    a = 2
    D = 1

    F_theta = torch.mean(MSE_theta)

    if flag == 0: mu_j = [(C*itr)**a for i in range(len(v_j))]
    else: mu_j = [D*itr*math.sqrt(itr) for i in range(len(v_j))]

    P_theta = [torch.mean(v_j[i]) for i in range(len(v_j))]
    P_theta = torch.divide(sum(P_theta), len(P_theta))

    if itr == 1:
        if P_theta <= tol:
            MSE_thetaf = (MSE_theta)
            F_thetaf = torch.mean(MSE_theta)
        else:
            MSE_thetaf = (torch.multiply(MSE_theta, 10.))
            F_thetaf = torch.mean(MSE_thetaf)


    if P_theta <= tol and torch.mean(MSE_theta) < torch.mean(MSE_thetaf):
            MSE_thetaf = MSE_theta
            F_thetaf = torch.mean((MSE_theta))

    #if P_theta <= tol:
    #    phi_theta = F_theta
    #else:
    phi_theta = F_theta + sum([torch.mul(mu_j[i], torch.divide(torch.sqrt(torch.sum((v_j[i]))), len(v_j[i]))) for i in range(len(v_j))])

    return phi_theta, MSE_thetaf



def lemongeAdaptive(MSE_theta, v_j, itr, MSE_thetaf):
    """
    Adaptation of the adaptive penalty function proposed in A. C. C. Lemonge and H. J. C. Barbosa, "An Adaptive Penalty Scheme for Genetic Algorithms in Structural Optimization"

    Args:
    MSE_theta (float): evaluation of the objective function f in the current point theta
    v_j (list of lists): list of constraints violations at each time step
    itr (int): optimization iteration number
    MSE_thetaf (float): value of objective function f if the point is feasible till this iteration

    Returns:
    phi_theta (float):
    MSE_thetaf (float):
    """
    tol = 1e-4

    F_theta = torch.mean(MSE_theta)

    mu_j = [torch.multiply(torch.mean(MSE_theta), torch.divide(torch.mean(v_j[i]), torch.square(torch.mean(v_j[i])))) for i in range(len(v_j))]
    P_theta = [torch.mean(v_j[i]) for i in range(len(v_j))]
    P_theta = torch.divide(sum(P_theta), len(P_theta))

    if itr == 1:
        if P_theta <= tol:
            MSE_thetaf = (MSE_theta)
            F_thetaf = torch.mean(MSE_theta)
        else:
            MSE_thetaf = (torch.multiply(MSE_theta, 10.))
            F_thetaf = torch.mean(MSE_thetaf)


    if P_theta <= tol and torch.mean(MSE_theta) < torch.mean(MSE_thetaf):
            MSE_thetaf = MSE_theta
            F_thetaf = torch.mean((MSE_theta))

    if P_theta <= tol:
        phi_theta = F_theta
    else:
        phi_theta = F_theta + sum([torch.mul(mu_j[i], torch.divide(torch.sum((v_j[i])), len(v_j[i]))) for i in range(len(v_j))])

    return phi_theta, MSE_thetaf


def selfAdaptive(MSE_theta, v_j, itr, MSE_thetaf):
    tol = 1e-4

    F_theta = torch.mean(normalizeFunc(MSE_theta))

    mu_j = [torch.div(torch.sum(v_j[i] != 0), len(v_j[i])) for i in range(len(v_j))]
    P_theta = [torch.mean(v_j[i]) for i in range(len(v_j))]
    P_theta = torch.divide(sum(P_theta), len(P_theta))

    if itr == 1:
        if P_theta <= tol:
            MSE_thetaf = MSE_theta
            F_thetaf = torch.mean(normalizeFunc(MSE_theta))
        else:
            MSE_thetaf = torch.multiply(MSE_theta, 10)
            F_thetaf = torch.mean(normalizeFunc(MSE_thetaf))


    if P_theta <= tol and torch.mean(MSE_theta) < torch.mean(MSE_thetaf):
            MSE_thetaf = MSE_theta
            F_thetaf = torch.mean(normalizeFunc(MSE_theta))

    if P_theta <= tol:
        phi_theta = F_theta
    else:
        phi_theta = F_theta + sum([torch.mul(mu_j[i], torch.divide(torch.sum(normalizeFunc(v_j[i])), len(v_j))) for i in range(len(v_j))])

    return phi_theta, MSE_thetaf

def normalizeFunc(x):
    return torch.subtract(1, torch.divide(1, torch.add(1, x)))
