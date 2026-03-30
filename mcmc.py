import emcee
from regression import get_A_from_k_ref, run_simulation, optimize_parameters
import tqdm
import corner
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import r2_score
from multiprocessing import Pool
import corner
import pickle

data_t_naoh = pickle.load(open('./data/data.pkl', 'rb'))
T_ref=70+273.15 # reference temperature for k_ref in K

x0 = [np.log(1e-2), 40, np.log(1e-1), 20, 0.2, 1.0, 1.0,]  # Initial guess for ln(k_ref_lig), Ea_lig, ln(k_ref_ace), Ea_ace, b_lig, n_lig, n_ace
lb = [np.log(1e-5), 20, np.log(1e-5), 0, 0.1, 0.5, 0.5]  # Lower bounds for ln(k_ref_lig), Ea_lig, ln(k_ref_ace), Ea_ace, b_lig, n_lig, n_ace
ub = [np.log(1e2), 100, np.log(1e2), 40, 0.8, 2.0, 2.0]  # Upper bounds for ln(k_ref_lig), Ea_lig, ln(k_ref_ace), Ea_ace, b_lig, n_lig, n_ace

sigma_lb = 0.001 # Minimum noise level (1% yield error)
sigma_ub = 0.1   # Maximum noise level (10% yield error)    
sigma_0 = 0.05   # Initial guess for noise level (5% yield error)

def log_prior(theta):
    # theta = [ln_k_ref_lig, Ea_lig, ln_k_ref_ace, Ea_ace, b_lignin, gamma, n, sigma]
    # Note: We add 'sigma' (noise level) as a parameter to be estimated!
    
    ln_k_lig, Ea_lig, ln_k_ace, Ea_ace, b_lig, gamma, n, sigma = theta
    
    # Check bounds (similar to your 'lb' and 'ub')
    if (lb[0] < ln_k_lig < ub[0] and lb[1] < Ea_lig < ub[1] and 
        lb[2] < ln_k_ace < ub[2] and lb[3] < Ea_ace < ub[3] and 
        lb[4] < b_lig < ub[4] and lb[5] <= gamma < ub[5] and 
        lb[6] < n < ub[6] and sigma_lb < sigma < sigma_ub):
        return 0.0 # uniform prior (equal probability within bounds)
    return -np.inf # impossible parameter set

def log_likelihood(theta, experimental_data):
    # Unpack parameters (excluding sigma for the simulation run)
    params_for_sim = theta[:-1] 
    sigma = theta[-1] # 
    
    residuals = run_simulation(params_for_sim, weights={'Lignin_yield': 1.0, 'Acetyl_yield': 1.0, 'NaOH_yield': 1.0}) 
    
    # Calculate Gaussian likelihood
    # Ln(L) = -0.5 * sum( (residual/sigma)^2 + ln(2*pi*sigma^2) )
    n_points = len(residuals)
    log_l = -0.5 * np.sum((residuals / sigma) ** 2 + np.log(2 * np.pi * sigma ** 2))
    return log_l

def log_probability(theta, experimental_data):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, experimental_data)

def run_mcmc(initial_guess, nwalkers=32, nsteps=5000):
    # Setup
    ndim = 8  # number of parameters (7 kinetic + 1 sigma)
    
    initial_guess_with_sigma = np.append(initial_guess, 0.05)
    pos = initial_guess_with_sigma + 1e-4 * np.random.randn(nwalkers, ndim)

    ncpu = os.cpu_count() - 2 
    print(f"Using {ncpu} CPUs")

    with Pool(processes=ncpu) as pool:
        
        # Pass the pool to the sampler
        sampler = emcee.EnsembleSampler(
            nwalkers, 
            ndim, 
            log_probability, 
            args=(data_t_naoh,), 
            pool=pool # <--- This is the magic argument
        )

        sampler.run_mcmc(pos, nsteps, progress=True)
    
    # Discard "burn-in" (the first 300 steps where it's finding the groove)
    flat_samples = sampler.get_chain(discard=400, thin=15, flat=True)
    return flat_samples, sampler

def run_all(chain_length=5000):
    results = optimize_parameters(x0, lb, ub)
    best_params = results.x
    print("Best parameters from optimization:", best_params)

    A_lig = get_A_from_k_ref(np.exp(best_params[0]), best_params[1], T_ref)
    A_ace = get_A_from_k_ref(np.exp(best_params[2]), best_params[3], T_ref)
    print(f"Optimized A_lig: {A_lig:.4e}, Ea_lig: {best_params[1]:.2f} kJ/mol, A_ace: {A_ace:.4e}, Ea_ace: {best_params[3]:.2f} kJ/mol, b_lig: {best_params[4]:.2e}, n_lig: {best_params[5]:.2f}, n_ace: {best_params[6]:.2f}")
    posterior_samples, sampler = run_mcmc(best_params, nwalkers=32, nsteps=chain_length)
    # Calculate the autocorrelation time (tau)
    tau = sampler.get_autocorr_time()
    print(f"Autocorrelation time for each parameter: {tau}")

    # Get the maximum tau across all your parameters
    max_tau = np.max(tau)

    # Calculate dynamic burn-in and thinning
    burnin = int(2 * max_tau)
    thinning = int(0.5 * max_tau) # take every Nth step to reduce file size/correlation

    print(f"burn in: {burnin} steps, thinning: every {thinning} steps")
    return posterior_samples, sampler

if __name__ == "__main__":
    # First, get a good initial guess from least_squares optimization
    results = optimize_parameters(x0, lb, ub)
    best_params = results.x
    print("Best parameters from optimization:", best_params)
    posterior_samples, sampler = run_mcmc(best_params)
    # Calculate the autocorrelation time (tau)
    tau = sampler.get_autocorr_time()
    print(f"Autocorrelation time for each parameter: {tau}")

    # Get the maximum tau across all your parameters
    max_tau = np.max(tau)

    # Calculate dynamic burn-in and thinning
    burnin = int(2 * max_tau)
    thinning = int(0.5 * max_tau) # take every Nth step to reduce file size/correlation

    print(f"burn in: {burnin} steps, thinning: every {thinning} steps")

    print("--- Final MCMC Parameters ---")
    labels = [
        r"$\ln(k_{ref, lig})$", 
        r"$E_{a, lig}$", 
        r"$\ln(k_{ref, ace})$", 
        r"$E_{a, ace}$", 
        r"$b_{lignin}$", 
        r"$n_{lignin}$", 
        r"$n_{ace}$", 
        r"$\sigma$ (noise)"
    ]
    # Iterate through each parameter
    for i in range(posterior_samples.shape[1]):
        # Calculate percentiles
        mcmc = np.percentile(posterior_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc) # Calculate differences: (50th - 16th) and (84th - 50th)
        
        # Identify the parameter name
        name = labels[i]
        
        # Text format: Value +UpperError -LowerError
        txt = f"{name}: {mcmc[1]:.3f} (+{q[1]:.3f} / -{q[0]:.3f})"
        print(txt)

        # Optional: If you want parameters for your simulation code
        if "lig" in name and "ln" in name:
            best_lnk_lig = mcmc[1]
        elif "Ea_lig" in name:
            best_Ea_lig = mcmc[1]

    labels = [
        r"$\ln(k_{ref, lig})$", 
        r"$E_{a, lig}$", 
        r"$\ln(k_{ref, ace})$", 
        r"$E_{a, ace}$", 
        r"$b_{lignin}$", 
        r"$n_{lignin}$", 
        r"$n_{ace}$", 
    ]

    # plt.rc('font', family='serif', serif='Times New Roman')
    # plt.rc('text', usetex=False)
    # plt.rc('xtick', labelsize=18)
    # plt.rc('ytick', labelsize=18)
    # plt.rc('axes', labelsize=18)
    import matplotlib as mpl
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = 'Times New Roman'

    plt.rcParams.update({'font.size': 10, 'font.family': 'serif', 'font.serif': 'Times New Roman'})

    # 2. Generate the plot
    # flat_samples is the result from sampler.get_chain(flat=True)
    fig = corner.corner(
        posterior_samples[:, :-1], 
        labels=labels, 
        show_titles=True,       
        quantiles=[0.16, 0.84],            # Used for the vertical dashed lines on the plot
        title_quantiles=[0.16, 0.5, 0.84], # REQUIRED: [16th, 50th, 84th] for "Median +Error -Error"
        title_fmt='.3f',
        label_kwargs={"fontsize": 18, "fontweight": 'bold', "fontname": "Times New Roman"}
    )
    # fig.subplots_adjust(right=0.95, top=0.95, hspace=0.1, wspace=0.1)
    for ax in fig.get_axes():
        ax.tick_params(axis='both', which='major', direction='out', length=6, width=1.5, labelsize=14)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.5)

    plt.savefig('./output/Corner_plot_MCMC_finer.png', dpi=300, bbox_inches='tight')
