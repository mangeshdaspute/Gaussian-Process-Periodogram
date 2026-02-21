import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import celerite
from celerite import terms
from joblib import Parallel, delayed
from tqdm import tqdm
from astropy.table import Table
from scipy.optimize import minimize_scalar


#####################calculate RMS of data. set upper bound for jitter as RMS and S0 using formula for S0 and lowest frequency. also reduce the n_points accordingly. It will increase computatiaonl efficiency. 

# Load data
name= 'activity RV asymmetric two spot configuration'
# Extract the time and radial velocity

df = pd.read_csv('timeseries activity RV asymmetric TWO spot region configuration.csv')

t = df['t']
y = df['y']
yerr = df['yerr']

plt.figure(figsize=(10, 6))
plt.errorbar(t-min(t), y, yerr=yerr, fmt=".k", capsize=0)
plt.xlabel("BJD-"+str(  min(t)  ))
plt.ylabel("RV [m/s]")
#plt.title(name)
plt.grid(True)
#plt.legend()
plt.savefig(name+' timeseries.png')
plt.show()
plt.close()


# Compute weights
weights = 1 / (yerr ** 2)
weighted_mean_value = np.average(y, weights=weights)
null_log_likelihood_value = np.sum(norm.logpdf(y, loc=weighted_mean_value, scale=yerr))
print('old weighted_mean_value = ',weighted_mean_value)
print('null_log_likelihood_value =', null_log_likelihood_value) 




def null_log_likelihood_with_jitter(ln_sigma_j_sq, y, yerr):
    """
    Calculates the maximum log-likelihood for the constant null model 
    given a specific jitter variance.
    
    Args:
        ln_sigma_j_sq (float): Log of the jitter variance (ln(sigma_j^2)).
        y (np.array): Array of data values.
        yerr (np.array): Array of measurement uncertainties (sigma_i).
        
    Returns:
        float: The negative maximum log-likelihood for that jitter variance.
               (We return the negative because we want to *minimize* this function).
    """
    # Exponentiate to get the jitter variance, ensuring it's non-negative
    sigma_j_sq = np.exp(ln_sigma_j_sq)
    
    # Calculate the total variance and weights
    total_variance = yerr**2 + sigma_j_sq
    weights = 1.0 / total_variance
    
    # Calculate the weighted mean (MLE of the constant mean mu)
    weighted_mean = np.average(y, weights=weights)
    
    # Calculate the log-likelihood
    log_likelihood = -0.5 * np.sum(
        (y - weighted_mean)**2 / total_variance + np.log(2 * np.pi * total_variance)
    )
    
    # The optimization function in scipy minimizes, so we return the negative log-likelihood
    return -log_likelihood

# --- Example Usage (Assuming y and yerr are defined) ---
# Example data (replace with your actual data)
# y = np.array([10.1, 10.5, 9.8, 11.0])
# yerr = np.array([0.5, 0.4, 0.3, 0.6])

# Find the jitter variance (sigma_j^2) that maximizes the log-likelihood.
# We optimize over ln(sigma_j^2) to ensure sigma_j^2 >= 0 and use a 
# reasonable search range for ln(sigma_j^2).
# A good starting range is from log(1e-10) up to log(max(yerr**2) * 10).
# The minimum jitter is sigma_j^2 = 0, which corresponds to ln(sigma_j^2) -> -inf.

# Set a very small minimum for the log-jitter-variance to approximate ln(0)
min_ln_sq = np.log(1e-10)
# Set a maximum log-jitter-variance (e.g., 10 times the max yerr^2)
max_ln_sq = np.log(10 * np.max(yerr**2)) 

# Perform the minimization
result = minimize_scalar(
    null_log_likelihood_with_jitter, 
    args=(y, yerr), 
    bounds=(min_ln_sq, max_ln_sq), 
    method='bounded'
)

# Extract the results
max_null_log_likelihood = -result.fun
optimal_sigma_j_sq = np.exp(result.x)
optimal_sigma_j = np.sqrt(optimal_sigma_j_sq)

print(f"Optimal Jitter Squared (sigma_j^2): {optimal_sigma_j_sq:.4f}")
print(f"Maximum Null Log Likelihood (with jitter): {max_null_log_likelihood:.4f}")

# You can also compute the final weighted mean using the optimal jitter
optimal_weights = 1.0 / (yerr**2 + optimal_sigma_j_sq)
weighted_mean_with_jitter = np.average(y, weights=optimal_weights)
print(f"Weighted Mean (with optimal jitter): {weighted_mean_with_jitter:.4f}")
weighted_mean_value =weighted_mean_with_jitter
null_log_likelihood_value = max_null_log_likelihood
# Compute w0 range
timespan_obs = np.max(t) - np.min(t)
w0min =  np.pi / timespan_obs
w0max = 2 * np.pi/1   # = np.pi
w0_list = np.arange(w0min, w0max, w0min)

# Grid bounds
log_S0_bounds = (-10, 5)
log_Q_lowest = np.log(0.502)
log_Q_bounds = (log_Q_lowest, 5)
log_sigma_bounds=(-10,5)
n_points = 7
log_S0_grid = np.linspace(log_S0_bounds[0], log_S0_bounds[1], n_points)
log_Q_grid = np.linspace(log_Q_bounds[0], log_Q_bounds[1], n_points)
log_sigma_grid = np.linspace(log_sigma_bounds[0], log_sigma_bounds[1], n_points)#sigma is jitter

# Fix log_sigma
#fixed_log_sigma = np.log(np.median(yerr))

def compute_for_w0(w0, t, y, yerr, weighted_mean_value, null_log_likelihood_value, log_S0_grid, log_Q_grid, log_sigma_grid):
    # Define kernel terms with current w0
    term1 = terms.SHOTerm(log_S0=0, log_Q=0, log_omega0=np.log(w0))
    term1.freeze_parameter("log_omega0")
    term2 = terms.SHOTerm(log_S0=0, log_Q=0, log_omega0=np.log(2 * w0))
    term2.freeze_parameter("log_omega0")
    jitter_term = terms.JitterTerm(log_sigma=-2)
    kernel = term1 + term2 + jitter_term
    gp = celerite.GP(kernel, mean=weighted_mean_value)
    gp.compute(t, yerr)
    
    # Initialize data dictionary for results
    data = {
        'w0': [],
        'log_S0_1': [],
        'log_Q1': [],
        'log_S0_2': [],
        'log_Q2': [],
        'log_sigma': [],
        'delta_ll': []
    }
    
    # Grid search over other parameters
    for log_S0_1 in log_S0_grid:
        for log_Q1 in log_Q_grid:
            for log_S0_2 in log_S0_grid:
                for log_Q2 in log_Q_grid:
                  for log_sigma in log_sigma_grid:
                    gp.set_parameter_vector([log_S0_1, log_Q1, log_S0_2, log_Q2, log_sigma])
                    log_likelihood = gp.log_likelihood(y)
                    delta_ll = log_likelihood - null_log_likelihood_value
                    data['w0'].append(w0)
                    data['log_S0_1'].append(log_S0_1)
                    data['log_Q1'].append(log_Q1)
                    data['log_S0_2'].append(log_S0_2)
                    data['log_Q2'].append(log_Q2)
                    data['log_sigma'].append(log_sigma)
                    data['delta_ll'].append(delta_ll)
    df = pd.DataFrame(data)
    optimal_df = df.loc[[df['delta_ll'].idxmax()]].reset_index(drop=True)
    return optimal_df

# Perform parallel grid search over w0
all_results = Parallel(n_jobs=-2)(delayed(compute_for_w0)(
    w0, t, y, yerr, weighted_mean_value, null_log_likelihood_value, log_S0_grid, log_Q_grid, log_sigma_grid
) for w0 in tqdm(w0_list, desc="Processing grid search of initial parameters without optimization")) 

# Combine all results into a single DataFrame
optimal_df = pd.concat(all_results, ignore_index=True)

# Save all results to CSV
#df.to_csv('grid_search_results'+name+'.csv', index=False)
#optimal_df['Q1'] = np.exp(optimal_df['log_Q1'])
#optimal_df['Q2'] = np.exp(optimal_df['log_Q2'])
#optimal_df['S0_1'] = np.exp(optimal_df['log_S0_1'])
#optimal_df['S0_2'] = np.exp(optimal_df['log_S0_2'])
#optimal_df['sigma'] = np.exp(optimal_df['log_sigma'])
optimal_df.to_csv('optimal_parameters '+name+'.csv', index=False)
# Plot optimal Q1 and Q2 against w0
plt.figure(figsize=(20, 8))
plt.plot(optimal_df['w0']/2/np.pi, np.exp(optimal_df['log_Q1']), 'o', label='Q1')
plt.plot(optimal_df['w0']/2/np.pi, np.exp(optimal_df['log_Q2']), 'o', label='Q2')
#plt.xscale('log')
plt.yscale('log')
plt.xlabel('Frequency [1/d]')
plt.ylabel('Q')
plt.legend()
plt.title('Optimal Q1 and Q2 vs frequency')
plt.savefig('Q_vs_frequency '+name+'.png')
plt.show()
plt.close()
 
