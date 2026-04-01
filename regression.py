from Deactylation import Deacetylation
from constants import *
import numpy as np
import matplotlib.pyplot as plt
import pickle

from scipy.optimize import least_squares


from timeit import Timer

data_t_naoh = pickle.load(open('./data/data.pkl', 'rb'))
T_ref=70+273.15 # reference temperature for k_ref in K


def get_k(k_ref, Ea, T_ref, T):
    k = k_ref * np.exp(-Ea*1000/R * (1/T - 1/T_ref))
    return k

def get_A_from_k_ref(k_ref, Ea, T_ref):
    A = k_ref / np.exp(-Ea*1000/(R*T_ref))
    return A

def run_simulation(params, weights={'Lignin_yield': 1.0, 'Acetyl_yield': 1.0}):
    ln_k_ref_lig, Ea_lig, ln_k_ref_ace, Ea_ace, b_lig, n_lig, n_ace = params
    k_ref_lig = np.exp(ln_k_ref_lig)
    k_ref_ace = np.exp(ln_k_ref_ace)

    all_residuals = []
    start = Timer().timeit()
    da = Deacetylation(silent=True) 
    for datadict in data_t_naoh:
        da.set_experimental_data(datadict)
        k_lig = get_k(k_ref_lig, Ea_lig, T_ref=T_ref, T=da._T)
        k_ace = get_k(k_ref_ace, Ea_ace, T_ref=T_ref, T=da._T)

        A = np.zeros(3)
        A[Lignin] = get_A_from_k_ref(k_ref_lig, Ea_lig, T_ref)
        A[Acetyl] = get_A_from_k_ref(k_ref_ace, Ea_ace, T_ref)
        Ea = np.zeros(3)
        Ea[Lignin] = Ea_lig*1000
        Ea[Acetyl] = Ea_ace*1000
        b = np.zeros(3)
        b[Lignin] = b_lig
        b[Acetyl] = 0.93 # theoretical acetyl to NaOH 1:1 molar ratio which converts to 0.93 g/g weight ratio based on molecular weights
        n = np.ones(3)
        n[Lignin] = n_lig
        n[Acetyl] = n_ace
        da.set_parameters(A, Ea, b, n)
        exp, pred = da.get_prediction()

        for key in ['Lignin_yield', 'Acetyl_yield']:
            residual = (np.array(exp[key]) - np.array(pred[key])) * weights[key]
            all_residuals.append(residual)

    return np.concatenate(all_residuals)

def run_simulation_raw(params, weights={'Lignin_yield': 1.0, 'Acetyl_yield': 1.0}):

    ln_k_ref_lig, Ea_lig, ln_k_ref_ace, Ea_ace, b_lig, n_lig, n_ace = params
    k_ref_lig = np.exp(ln_k_ref_lig)
    k_ref_ace = np.exp(ln_k_ref_ace)

    lignin_residuals = []
    acetyl_residuals = []
    naoh_residuals = []
    start = Timer().timeit()
    da = Deacetylation(silent=True) 
    for datadict in data_t_naoh:
        da.set_experimental_data(datadict)
        k_lig = get_k(k_ref_lig, Ea_lig, T_ref=T_ref, T=da._T)
        k_ace = get_k(k_ref_ace, Ea_ace, T_ref=T_ref, T=da._T)

        A = np.zeros(5)
        A[Lignin] = get_A_from_k_ref(k_ref_lig, Ea_lig, T_ref)
        A[Acetyl] = get_A_from_k_ref(k_ref_ace, Ea_ace, T_ref)
        Ea = np.zeros(5)
        Ea[Lignin] = Ea_lig*1000
        Ea[Acetyl] = Ea_ace*1000
        b = np.zeros(5)
        b[Lignin] = b_lig
        b[Acetyl] = 0.93 # theoretical acetyl to NaOH 1:1 molar ratio which converts to 0.93 g/g weight ratio based on molecular weights
        n = np.ones(5)
        n[Lignin] = n_lig
        n[Acetyl] = n_ace
        da.set_parameters(A, Ea, b, n)
        exp, pred = da.get_prediction()

        lignin_res = (np.array(exp['Lignin_yield']) - np.array(pred['Lignin_yield'])) * weights['Lignin_yield']
        acetyl_res = (np.array(exp['Acetyl_yield']) - np.array(pred['Acetyl_yield'])) * weights['Acetyl_yield']
        lignin_residuals.append(lignin_res)
        acetyl_residuals.append(acetyl_res)

    return np.concatenate(lignin_residuals), np.concatenate(acetyl_residuals)

def optimize_parameters(x0, lb, ub):
    result = least_squares(run_simulation, x0=x0, bounds=(lb, ub), method='trf', loss='soft_l1', f_scale=1, verbose=2)
    return result

def compute_correlation_matrix(result, name_list=['ln_k_ref_lig', 'Ea_lig', 'ln_k_ref_ace', 'Ea_ace', 'b_lig', 'n_lig', 'n_ace']):
    # Extract the Jacobian matrix at the optimized solution
    J = result.jac
    # Calculate the inverse of the approximated Hessian (J^T * J)^-1
    # This gives the unscaled covariance matrix
    JTJ_inv = np.linalg.inv(J.T @ J)
    # Calculate the residual variance (Mean Squared Error)
    n = len(result.fun) # Number of data points (residuals)
    p = len(result.x)   # Number of optimized parameters
    mse = np.sum(result.fun**2) / (n - p)
    # Calculate the true Covariance Matrix
    cov_matrix = mse * JTJ_inv
    # Convert the Covariance Matrix to a Correlation Matrix
    # Extract the standard deviations (square root of the diagonal elements)
    std_devs = np.sqrt(np.diag(cov_matrix))
    # Compute the correlation matrix: C_ij = Cov_ij / (std_i * std_j)
    std_matrix = np.outer(std_devs, std_devs)
    correlation_matrix = cov_matrix / std_matrix
    # print the correlation matrix with parameter names
    print("Correlation Matrix:")
    print(" " * 2 + " ".join(f"{name:>3} & " for name in name_list))
    for i, name in enumerate(name_list):
        row = " ".join(f"{correlation_matrix[i, j]:3.4f} & " for j in range(len(name_list)))
        print(f"{name:>15} & {row}")  

        
if __name__ == "__main__":
    x0 = [np.log(1e-2), 40, np.log(1e-1), 20, 0.2, 1.0, 1.0]  # Initial guess for ln(k_ref_lig), Ea_lig, ln(k_ref_ace), Ea_ace, b_lig, n_lig, n_ace
    lb = [np.log(1e-5), 20, np.log(1e-5), 0, 0.1, 0.5, 0.5]  # Lower bounds for ln(k_ref_lig), Ea_lig, ln(k_ref_ace), Ea_ace
    # ub = [np.log(1e2), 100, np.log(1e2), 40, 0.5, 2.0, 2.0]  # Upper bounds for ln(k_ref_lig), Ea_lig, ln(k_ref_ace), Ea_ace
    ub = [np.log(1e2), 100, np.log(1e2), 40, 0.8, 2.0, 2.0]  # Upper bounds for ln(k_ref_lig), Ea_lig, ln(k_ref_ace), Ea_ace
    results = optimize_parameters(x0, lb, ub)
    best_params = results.x
    print("Best parameters found: ", best_params)
    compute_correlation_matrix(results)

    A_lig = get_A_from_k_ref(np.exp(best_params[0]), best_params[1], T_ref)
    A_ace = get_A_from_k_ref(np.exp(best_params[2]), best_params[3], T_ref)
    print(f"Optimized A_lig: {A_lig:.4e}, Ea_lig: {best_params[1]:.2f} kJ/mol, A_ace: {A_ace:.4e}, Ea_ace: {best_params[3]:.2f} kJ/mol, b_lig: {best_params[4]:.2e}, n_lig: {best_params[5]:.2f}, n_ace: {best_params[6]:.2f}")
