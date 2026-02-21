#Q, lifetime and period calculated for GJ 880 = 0.939, 6, 20 are same by both grid search and default gp
#try lsgb method of optimizatoin on undersampling  
#decay lifetime is correct.
#limit upper bound on S0

import copy
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import norm
import celerite
from celerite import terms
import time
from joblib import Parallel, delayed
import matplotlib.colors as mcolors
from tqdm import tqdm
import math
import mlp
from astropy.timeseries import LombScargle
import random
from astropy.table import Table
from matplotlib import gridspec

name= 'activity RV asymmetric two spot region configuration'

optimal_parameters_filename='optimal_parameters timeseries activity RV asymmetric TWO spot region configuration'

df = pd.read_csv('timeseries activity RV asymmetric TWO spot region configuration.csv')
t = df['t']
y = df['y']
yerr = df['yerr'] 

dt = t.diff().dropna() 

median_dt = np.median(dt)
plt.figure(figsize= (10,6), dpi=300)
plt.plot(dt,'r.')
plt.yscale('log')
plt.ylabel('dt [d]')
plt.xlabel('index of observation')
#plt.title(name +'median dt = '+str(median_dt*24)+' hours')
plt.grid(True)
plt.show()
plt.savefig(name+' dt.png')
print('median dt = ', median_dt)
print('median uncertainity = ', np.median(yerr))


plt.figure(figsize=(10,6), dpi=300)
plt.plot(yerr, 'r.')
plt.xlabel('index of observation')
plt.ylabel('RV uncertainity [m/s]')
#plt.title(name)
plt.grid(True)
plt.savefig(name+' uncertainity.png')

# Compute residuals (use mean RV as the model baseline)
model = np.mean(y)	
residuals = y - model
rms_scatter = np.sqrt(np.mean((y - np.mean(y))**2))
print("rms_scatter before clipping = ",rms_scatter)
# Standardize residuals
standardized_residuals = residuals / yerr


print('length of t = ',len(t))
plt.figure(figsize=(10, 6), dpi=300)
plt.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0, label= 'RMS scatter ='+f"{rms_scatter:.3f} m/s")
plt.xlabel("Time [d]")
plt.ylabel("RV [m/s]")
#plt.title(name)
plt.grid(True)
plt.legend()
plt.savefig(name+' timeseries.png')
plt.show(block=False)
plt.close()



# Compute the weighted mean directly using numpy
weights = 1 / yerr**2
weighted_mean_value = np.average(y, weights=weights)  # Weighted mean
#####################Subtracting by weighted mean to have mean of data = 0 #########################
#y=y-weighted_mean_value
#weighted_mean_value=0
print("type of y = ", type(y))
# Use scipy.stats.norm.logpdf to compute log-likelihood
null_log_like = np.sum(norm.logpdf(y, loc=weighted_mean_value, scale=yerr))

print(f"Weighted Mean: {weighted_mean_value:.4f}")
print(f"Log-Likelihood: {null_log_like:.4f}")


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
    log_likelihood = -0.5 * np.sum(        (y - weighted_mean)**2 / total_variance   +   np.log(2 * np.pi * total_variance)       )
    
    # The optimization function in scipy minimizes, so we return the negative log-likelihood
    return -log_likelihood

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

result = minimize_scalar(
    null_log_likelihood_with_jitter, 
    args=(y, yerr), 
    bounds=(min_ln_sq, max_ln_sq), 
    method='bounded'
)

# Extract the results
null_log_like = -result.fun
optimal_sigma_j_sq = np.exp(result.x)
optimal_sigma_j = np.sqrt(optimal_sigma_j_sq)
jitter= 1.0          ###########################################################optimal_sigma_j
print(f"Optimal Jitter Squared (sigma_j^2): {optimal_sigma_j_sq:.4f}")
print(f"Maximum Null Log Likelihood (with jitter): {max_null_log_likelihood:.4f}")

# You can also compute the final weighted mean using the optimal jitter
optimal_weights = 1.0 / (yerr**2 + optimal_sigma_j_sq)
weighted_mean_value = np.average(y, weights=optimal_weights)
print(f"Weighted Mean (with optimal jitter): {weighted_mean_with_jitter:.4f}")


timespan_obs=max(t)-min(t)
print("timespan_obs = ",timespan_obs)
w0min=np.pi/(5*timespan_obs)

# Compute differences and drop the first NaN

print("median dt = ",median_dt)
w0max=2*np.pi/1  #frequency>2  shows reflection of periodogram at frequency=1/2
#w0 points = 10
w0_list=np.arange(w0min, w0max, w0min)
print('length of w0_list = ',len( w0_list))
Qmin=0.5  #Q<0.5 means overdamping
Qmax=10    #Q= 10 is almost undamped
#Q points= 20
l_list=np.arange(1/Qmax,1/Qmin,1/Qmax/10.0)
#print(len(l_list))
frequency = w0_list/(2*np.pi )
fbeg=frequency[0]
fend = frequency[-1]
data_tuple = (t, y, yerr)


# Compute GLS periodogram
ls = LombScargle(t, y, yerr)
power = ls.power(frequency)
#jitt = ls.jitter
# Convert frequency to period
#periods = 1 / frequency

# Compute the power level for 1% FAP
fap_level = 0.01
power_fap = ls.false_alarm_level(0.01)
power_fap2 = ls.false_alarm_level(0.1)
# Plot the periodogram
plt.figure(figsize=(10, 6), dpi=300)
plt.plot(frequency, power, color='black', lw=0.5)
plt.axhline(power_fap, color='blue', linestyle='-', label=f'1% FAP ({power_fap:.2f})')
plt.axhline(power_fap2, color='green', linestyle='--', label=f'10% FAP ({power_fap2:.2f})')
plt.xlabel('frequency [1/days]')
plt.ylabel('Power')
#plt.title('GLS Periodogram')
plt.legend()
plt.grid(True)
plt.savefig('GLS astropy '+name+'.png')
plt.show(block=False)
plt.close()

gls = mlp.Gls([data_tuple],  fbeg=fbeg, fend=fend, norm='dlnL', freq=frequency, verbose=True)


fap_level = 0.001  # 1% FAP
#lnL_threshold = gls.powerLevel(fap_level)
#dlnL_threshold = lnL_threshold - gls.lnML0

def bootstrap_fap(t, y, yerr, freq, n_boot):
    def single_bootstrap():
        #y_shuffled = np.random.permutation(y)
        y_simulated=0#np.random.normal(0, yerr, len(t))
        #np.random.seed(42)  # For reproducibility, optional
        noise = np.random.normal(loc=0, scale=np.sqrt(yerr**2+jitter**2), size=t.shape)
        y_shuffled= y_simulated+noise
        gls_sim = mlp.Gls([(t, y_shuffled, yerr)], fbeg=freq[0], fend=freq[-1], freq=freq, norm="dlnL")
        return gls_sim.power.max()  # For dlnL, power is already Δ lnL

    # Parallelize bootstrap iterations with tqdm
    max_dlnL = Parallel(n_jobs=-2)(
        delayed(single_bootstrap)() for _ in tqdm(range(n_boot), desc="GLS Bootstrap FAP")
    )
    return np.percentile(max_dlnL, [ 99, 90])  # thresholds for FAP=0.001, 0.01, 0.1

# Compute FAP thresholds using bootstrap (single call)
FAP_result = bootstrap_fap(t, y, yerr, frequency, n_boot=100)
#power_threshold_001 = FAP_result[0]  # FAP = 0.0001
power_threshold_1p = FAP_result[0]  # FAP = 0.001
power_threshold_10p =FAP_result[1]  # FAP = 0.01
#power_threshold_1 = FAP_result[2]  # FAP = 0.1
plt.figure(figsize=(10, 6), dpi=200)
plt.plot(gls.freq, gls.power, 'k-')
#plt.axhline(y=power_threshold_10p, color='r', linestyle='--', label='0.1 FAP')
#plt.axhline(y=power_threshold_1p, color='b', linestyle='--', label='0.01 FAP')

#plt.axhline(y=power_threshold_1, color='r', linestyle='-.')
#plt.axhline(y=power_threshold_1, color='r', linestyle=':')
plt.xlabel("Frequency [1/d]")
plt.ylabel("Δ lnL")
#plt.title(" GLS Periodogram"+name)
#plt.xscale("log")
#plt.gca().invert_xaxis()  # Optional: invert x-axis so shorter periods are on the right
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.xlim([0,1])
#plt.legend()
plt.savefig("GLS "+name+" delta log likelihood peridogram.png")
plt.show(block=False)
plt.close()


plt.figure(figsize=(10, 6), dpi=300)
plt.plot(gls.freq, gls.power, 'k-')
#plt.axhline(y=power_threshold_1p, color='b', linestyle='-', label='1% FAP')
#plt.axhline(y=power_threshold_10p, color='g', linestyle='--', label='10% FAP')
#plt.axhline(y=power_threshold_1, color='r', linestyle='-.')
#plt.axhline(y=power_threshold_1, color='r', linestyle=':')
plt.xlabel("Frequency [1/d]")
plt.ylabel("Δ lnL")
#plt.title(" GLS Periodogram"+name+" zoomed in")
#plt.xscale("log")
#plt.gca().invert_xaxis()  # Optional: invert x-axis so shorter periods are on the right
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.xlim([0,0.1])
plt.legend()
plt.savefig("GLS "+name+" delta log likelihood peridogram zoomed in.png")
plt.show(block=False)
plt.close()


# Assuming gls is already computed
offsets = gls._off[:, 0]  # Flatten to 1D array for single series
jitters = np.array([p[-gls.Nj:] for p in gls.par])[:, 0]  # Flatten to 1D
amplitudes = np.sqrt(gls._a**2 + gls._b**2)

# Plot frequency vs offset
plt.figure(figsize=(10, 6), dpi=300)
plt.plot(gls.freq, offsets)
plt.xlabel('Frequency [1/d]')
plt.ylabel('Offset [m/s]')
plt.title('Frequency vs Offset '+name)
plt.savefig("GLS "+name+" offsets.png")
plt.show(block=False)
plt.close()
# Plot frequency vs jitter
plt.figure(figsize=(10, 6), dpi=300)
plt.plot(gls.freq, jitters)
plt.xlabel('Frequency [1/d]')
plt.ylabel('Jitter [m/s]')
plt.title('Frequency vs Jitter '+name)
plt.savefig("GLS "+name+" jitters.png")
plt.show(block=False)
plt.close()

# Create DataFrame
df_mlp_gls = pd.DataFrame({
    'frequency': gls.freq,
    'jitter': jitters,
    'delta_log_likelihood': gls.power,
    'offset': offsets,
    'amplitudes' : amplitudes

})
df_mlp_gls.to_csv('gls_mlp_results'+name+'.csv', index=False)

bounds = dict(log_S0=(-15, 15), log_Q=(-15, 15), log_omega0=(-15, 15))
#kernel = terms.SHOTerm(log_S0=np.log(S0), log_Q=np.log(Q), log_omega0=np.log(w0),
                       #bounds=bounds)
#kernel.freeze_parameter("log_Q")  # We don't want to fit for "Q" in this term

# A periodic component
Q = 1.20
w0 = 3.0
S0 = np.var(y) / (w0 * Q)
offset=0
kernel = terms.SHOTerm(log_S0=np.log(S0), log_Q=np.log(Q), log_omega0=np.log(w0),
                        bounds=bounds)
gp = celerite.GP(kernel, mean=weighted_mean_value )#+ offset, fit_mean=True
#gp = celerite.GP(kernel, mean=weighted_mean_value)
gp.compute(t, yerr)  # You always need to call compute once.
print("Initial log likelihood: {0}".format(gp.log_likelihood(y)))

def neg_log_like(params, y, gp):
    gp.set_parameter_vector(params)
    return -gp.log_likelihood(y)

initial_params = gp.get_parameter_vector()
bounds = gp.get_parameter_bounds()

r = minimize(neg_log_like, initial_params, method="L-BFGS-B", bounds=bounds, args=(y, gp))
gp.set_parameter_vector(r.x) #update gp with optimized parameters
#print(r)
gp_original=gp
print(gp_original.log_likelihood(y)-null_log_like)
print(gp_original.get_parameter_dict())


x = np.linspace(min(t), max(t), 5000)
pred_mean, pred_var = gp_original.predict(y, x, return_var=True)
pred_std = np.sqrt(pred_var)
color = "#ff7f0e"
#plt.plot(true_t, true_y, "k", lw=1.5, alpha=0.3)
plt.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0)
plt.plot(x, pred_mean, color=color)
plt.fill_between(x, pred_mean+pred_std, pred_mean-pred_std, color=color, alpha=0.3,
                 edgecolor="none")
plt.xlabel("x")
plt.ylabel("y")
#plt.xlim(0, 50)
plt.title('data vs prediction using SHO kernel')

plt.show(block=False)
plt.close()
#plt.ylim(-2.5, 2.5);


# Define your variables, inputs, and functions as needed
#bounds = dict(log_S0=(-15, 15), log_Q=(-15, 15), log_omega0=(-15, 15))
#offset = 0
start_time = time.perf_counter()
count=0
# Function to process a single w0 valu

def process_w0(w0, y, t, yerr, timespan_obs, weighted_mean_value, null_log_like,df):
    # Initial parameter guesses
    w0copy=w0
    if 'w0' not in df.columns:
        raise ValueError("CSV file must contain 'w0'")
    closest_w0_idx = (df['w0'] - w0).abs().idxmin()
    filtered_df = df[df['w0'] == w0]
    print('w0 target = ',w0)
    w0 = df.loc[closest_w0_idx, 'w0']
    print('w0 selected from csv = ',w0)
    log_S0_1 = df.loc[closest_w0_idx, 'log_S0_1']
    log_Q1 = df.loc[closest_w0_idx, 'log_Q1']
    log_S0_2 = df.loc[closest_w0_idx, 'log_S0_2']
    log_Q2 = df.loc[closest_w0_idx, 'log_Q2']
    log_sigma = df.loc[closest_w0_idx, 'log_sigma']
    w0=w0copy
    
    log_Q_lowest = np.log(0.501)

    # Define parameter bounds
    bounds = {"log_S0": (-15, 15), "log_Q": (log_Q_lowest, 15), "log_omega0": (-15, 15)}
    # Define two-term kernel: one for w0, one for w1 = 2*w0
     
    term1 = terms.SHOTerm(log_S0=log_S0_1, log_Q=log_Q1,  log_omega0=np.log(w0), bounds=bounds)
    #log_w0_min=np.log(2 * w0*0.95)
    #log_w0_max=np.log(2 * w0*1.05)
    bounds2 = {"log_S0": (-15, 15), "log_Q": (log_Q_lowest, 15), "log_omega0": (-15, 15)}
    term2 = terms.SHOTerm(log_S0=log_S0_2, log_Q=log_Q2, log_omega0=np.log(2 * w0), bounds=bounds2)
    
    
    # Freeze log_omega0 parameters since they are fixed
    term1.freeze_parameter("log_omega0")
    term2.freeze_parameter("log_omega0")
    jitter_term = terms.JitterTerm( log_sigma=log_sigma, bounds={"log_sigma": (-15, 15)} )  # Adjust
    
    kernel = term1 + term2+ jitter_term
    # Set up Gaussian Process
    #mean_model = models.ConstantModel(value=weighted_mean_value)
    #mean_model = modeling.ConstantModel(weighted_mean_value)
    gp = celerite.GP(kernel, mean=weighted_mean_value)
    gp.compute(t, yerr)
    
    # Define negative log-likelihood function
    def neg_log_like(params, y, gp):
        gp.set_parameter_vector(params)
        return -gp.log_likelihood(y)
    
    # Get initial parameters and set bounds
    initial_params = gp.get_parameter_vector()
    #bounds = [(-15, 15)] * len(initial_params)
    
    # Optimize parameters
    r = minimize(neg_log_like, initial_params, method="L-BFGS-B", bounds=gp.get_parameter_bounds(), args=(y, gp))
    gp.set_parameter_vector(r.x)
    
    # Extract optimized parameters
    optimized_params = gp.get_parameter_dict()
    print('optimized_params dictionary= ', optimized_params)
    S0_1 = np.exp(optimized_params['kernel:terms[0]:log_S0'])
    Q_1 = np.exp(optimized_params['kernel:terms[0]:log_Q'])
    S0_2 = np.exp(optimized_params['kernel:terms[1]:log_S0'])
    Q_2 = np.exp(optimized_params['kernel:terms[1]:log_Q'])
    w_2 = 2*w0#
    jitter=np.exp(optimized_params['kernel:terms[2]:log_sigma'])
    #offset = optimized_params['mean:value']

    #print(f"Optimized Offset: {offset}")
    #you should not keep w_2 free to optimize beyound 5% of 2*w0 (0.5 time the period). Because then it prioretizes optimiing double the period of rotation of star.
     
    delta_log_like = gp.log_likelihood(y) - null_log_like
    
    # Return results as a single list
    return [S0_1, S0_2, Q_1, Q_2, delta_log_like,w0,w_2,jitter,null_S0_1, null_S0_2, null_Q_1, null_Q_2,null_jitter,null_log_like] # offset
# Parallel execution
#seasonal data plot
# Starting point
start_t = 2457500
days_in_section = 365

t_original=t
y_original=y
yerr_original=yerr
# Create the 9 sections
sections = []
for i in range(1):
    # Define the start and end times for the current section
    start_time = start_t + i * days_in_section
    end_time = start_t + (i + 1) * days_in_section

    # Filter the data for the current section
    mask = (t_original >= start_time) & (t_original < end_time)
    #t = t_original[mask]
    #y = y_original[mask]
    #yerr = yerr_original[mask]

    # Store the section
    sections.append((t, y, yerr))
    # Compute the weighted mean directly using numpy
    weights = 1 / yerr**2
    weighted_mean_value = np.average(y, weights=weights)  # Weighted mean
    #####################Subtracting by weighted mean to have mean of data = 0 #########################
    #y=y-weighted_mean_value
    #weighted_mean_value=0
    print("type of y = ", type(y))
    # Use scipy.stats.norm.logpdf to compute log-likelihood
    #null_log_like = np.sum(norm.logpdf(y, loc=weighted_mean_value, scale=yerr))
    result = minimize_scalar(
        null_log_likelihood_with_jitter, 
        args=(y, yerr), 
        bounds=(min_ln_sq, max_ln_sq), 
        method='bounded'
    )

    # Extract the results
    null_log_like = -result.fun
    optimal_sigma_j_sq = np.exp(result.x)
    optimal_sigma_j = np.sqrt(optimal_sigma_j_sq)

    print(f"Optimal Jitter Squared (sigma_j^2): {optimal_sigma_j_sq:.4f}")
    print(f"Maximum Null Log Likelihood (with jitter): {max_null_log_likelihood:.4f}")

    # You can also compute the final weighted mean using the optimal jitter
    optimal_weights = 1.0 / (yerr**2 + optimal_sigma_j_sq)
    weighted_mean_value = np.average(y, weights=optimal_weights)
    print(f"Weighted Mean (with optimal jitter): {weighted_mean_with_jitter:.4f}")
    

    timespan_obs=max(t)-min(t)
    w0min=np.pi/(timespan_obs)
    w0max=2*np.pi/1   #frequency>2  shows reflection of periodogram at frequency=1/2
    #w0 points = 10
    w0_list=np.arange(w0min, w0max, w0min)
    df = pd.read_csv(optimal_parameters_filename+'.csv')
    print("Columns in df:", df.columns.tolist())
    print("Shape of df:", df.shape)
    print("Sample row:\n", df.iloc[0])
    df_copy=df
    #max_row = filtered_df.loc[filtered_df['delta_ll'].idxmax()]
    #w0_list = df['w0'].unique()

    result = Parallel(n_jobs=-2)  (delayed(process_w0)(w0, y, t, yerr, timespan_obs, weighted_mean_value, null_log_like,df) for w0 in tqdm(w0_list, desc="Processing grid search")) 

    # Flatten the results list
    #result = [item for sublist in results for item in sublist]
    print("type of result",type(result))
    # Create a DataFrame
    df_results = pd.DataFrame(result, columns=['S0_1', 'S0_2', 'Q_1', 'Q_2', 'delta_log_like', 'w0', 'w_2','jitter','null_S0_1', 'null_S0_2', 'null_Q_1', 'null_Q_2','null_jitter','null_log_like'])
    df_results['Frequency_1'] =   df_results['w0']/(2 * np.pi)
    df_results['Frequency_2'] = df_results['w_2']/(2 * np.pi)
    df_results['Period_1'] = 2 * np.pi / df_results['w0']
    df_results['Period_2'] = 2 * np.pi / df_results['w_2']
    #df_results['offset'] = 2 * np.pi / df_results['offset']
    #############lifetime formula is correct###############
    df_results['lifetime_1'] = 2 * df_results['Q_1'] / df_results['w0']
    df_results['lifetime_2'] = 2 * df_results['Q_2'] / df_results['w_2']
    #############amplitude formula is correct###############
    df_results['Amplitude_1[m/s]'] = np.sqrt(2*df_results['Q_1'] * df_results['w0']*df_results['S0_1'])
    df_results['Amplitude_2[m/s]'] = np.sqrt(2*df_results['Q_2'] * df_results['w_2']*df_results['S0_2'])
    df_results['fraction_RMS1[m/s]'] =df_results['Amplitude_1[m/s]']/(     math.sqrt(2)*rms_scatter    )
    df_results['fraction_RMS2[m/s]'] = df_results['Amplitude_2[m/s]']/(     math.sqrt(2)*rms_scatter    )
    # Save to CSV
    df_results.to_csv(name+'_optimized_1D_GP_periodogram.csv', index=False)


    # Extract highest delta_log_likelihood_solution
    highest_delta_log_likelihood_solution = max(result, key=lambda item: item[4])
    #jitter = highest_delta_log_likelihood_solution[7]
    #print("type of highest_delta_log_likelihood_solution[1]",type(highest_delta_log_likelihood_solution[1]))
    #print("type of highest_delta_log_likelihood_solution ",type(highest_delta_log_likelihood_solution))
    #highest_delta_log_likelihood_solution[1] = float(2*np.pi / highest_delta_log_likelihood_solution[1])   ##############causes contour plot to change##########
    #highest_delta_log_likelihood_solution[4] = float(1 / highest_delta_log_likelihood_solution[4])##############causes contour plot to change########
#    formatted_highest_delta_log_likelihood_solution = [f"{x:.5f}" for x in highest_delta_log_likelihood_solution]
#    opt_Q=highest_delta_log_likelihood_solution[2]
#    opt_w0=highest_delta_log_likelihood_solution[1]
#    opt_S0=highest_delta_log_likelihood_solution[0]
#    opt_kernel =  terms.SHOTerm(log_S0=np.log(opt_S0), log_Q=np.log(opt_Q), log_omega0=np.log(opt_w0))
#    opt_gp = celerite.GP(opt_kernel)
#    opt_gp.compute(t, yerr)  # Compute the covariance matrix
#    t_pred = np.linspace(min(t), max(t), 5000)  # Fine grid for prediction
#    pred_mean, pred_var = opt_gp.predict(y, t_pred, return_var=True)
#    pred_std = np.sqrt(pred_var)
#    # Plot results
#    plt.figure(figsize=(10, 6))
#    plt.errorbar(t, y, yerr=yerr, fmt=".k", label="Observations", capsize=2)
#    plt.plot(t_pred, pred_mean, "b", label="GP Prediction")
#    plt.fill_between(t_pred, pred_mean - pred_std, pred_mean + pred_std,color="blue", alpha=0.2, label="1σ uncertainty")
#    plt.xlabel("Time")
#    plt.ylabel("Observation")
#    plt.legend()

#    plt.title("GP Prediction heighest delta likelyhood solution GJ 686 Optimized parameters \n [S0, Angular frequency, Q, delta_log_like, 1/lifetime]"+str(formatted_highest_delta_log_likelihood_solution))
#    plt.savefig("GP prediction heighest delta likelyhood solution GJ 686"+str(formatted_highest_delta_log_likelihood_solution)+".png")
#    plt.show()
    
    
    ##################select only top 500 highest delta_log_likelyhood solutions for better visualisation#######################
    #top_50_solutions = sorted(result, key=lambda item: item[3], reverse=True)[:500]
    #print('type of top_50_solutions',type(top_50_solutions))
    #result=top_50_solutions
    # Extract variables from result
    S0_values = [i[0] for i in result]
    S2_values = [i[1] for i in result]
    w0_values = [i[5] for i in result]
    w2_values = [i[6] for i in result]
    Q0_values = [i[2] for i in result]
    Q2_values = [i[3] for i in result]
    delta_log_like = [i[4] for i in result]
    null_S0_values = [i[8] for i in result]
    null_S2_values = [i[9] for i in result]
    null_Q0_values = [i[10] for i in result]
    null_Q2_values = [i[11] for i in result]
    null_log_like = [i[13] for i in result]
    jitter_values = [i[7] for i in result]
    null_jitter_values = [i[12] for i in result]

    lifetime0_values =  2*np.array(Q0_values)/np.array(w0_values)#2*np.pi/w0_values*Q0_values
    lifetime2_values = 2*np.array(Q2_values)/np.array(w2_values)

    null_lifetime0_values =  2*np.array(null_Q0_values)/np.array(w0_values)#2*np.pi/w0_values*Q0_values
    null_lifetime2_values = 2*np.array(null_Q2_values)/np.array(w2_values)
    # Frequency calculations
    Frequency_val = np.array(w0_values) / (2 * np.pi)
    Frequency_val2 = np.array(w2_values) / (2 * np.pi)

    plt.figure(figsize =(10,6), dpi=200 )
    plt.plot(Frequency_val, delta_log_like, color='gray', linewidth=0.5, alpha=0.5, zorder=1)
    sc = plt.scatter(Frequency_val, delta_log_like, 
                 c=df_results['fraction_RMS1[m/s]'], 
                 cmap='rainbow', 
                 s=10, 
                 edgecolor='none', 
                 zorder=2)
    # 2. Add a colorbar to explain what the colors represent
    cbar = plt.colorbar(sc)
    cbar.set_label('RMS fraction of first oscillator')
    plt.xlabel('Frequency of first oscillator [1/d]')
    plt.ylabel('Δ lnL')
    #plt.title('1D GP Periodogram '+name)
    
    plt.xlim([0,1])
    plt.grid(which='major', linestyle='-')
    plt.xlim([0.0,1.0])
    #plt.xscale('log')
    #plt.minorticks_on()  # This enables minor ticks
    plt.grid(which='minor', color='#DDDDDD', linestyle='--', alpha=0.6)
    plt.savefig("GP 1D double sho "+name+" after gridsearch on initial parameters.png")
    plt.show()    



    ####zoomed in   single colour
    plt.figure(figsize =(10,6), dpi=200 )
    plt.plot(Frequency_val,delta_log_like,'r')
    plt.xlabel('Frequency of first oscillator [1/d]')
    plt.ylabel('Δ lnL')
    #plt.title('1D GP Periodogram '+name)
    
    plt.ylim([np.median(delta_log_like)/2,max(delta_log_like)])
    plt.grid(which='major', linestyle='-')
    plt.xlim([0,0.1])
    #plt.xscale('log')
    plt.minorticks_on()  # This enables minor ticks
    plt.grid(which='minor', color='#DDDDDD', linestyle='--', alpha=0.6)
    plt.savefig("GP 1D double sho "+name+" after gridsearch on initial parameters zoomed in.png")
    plt.show()


    ################ GP periodogram zoomed in multicolour
    plt.figure(figsize =(10,6), dpi=200 )
    plt.plot(Frequency_val, delta_log_like, color='gray', linewidth=0.5, alpha=0.5, zorder=1)
    sc = plt.scatter(Frequency_val, delta_log_like, 
                 c=df_results['fraction_RMS1[m/s]'], 
                 cmap='rainbow', 
                 s=10, 
                 edgecolor='none', 
                 zorder=2)
    # 2. Add a colorbar to explain what the colors represent
    cbar = plt.colorbar(sc)
    cbar.set_label('RMS fraction of first oscillator')
    plt.xlabel('Frequency of first oscillator [1/d]')
    plt.ylabel('Delta log likelihood')
    #plt.title('1D GP Periodogram '+name)
    
    #plt.ylim([30,60])
    plt.grid(which='major', linestyle='-')
    plt.xlim([0.0,0.1])
    #plt.xscale('log')
    plt.minorticks_on()  # This enables minor ticks
    plt.grid(which='minor', color='#DDDDDD', linestyle='--', alpha=0.6)
    plt.savefig("GP 1D double sho "+name+" after gridsearch on initial parameters zoomed in.png")
    plt.show()    
    
    ############################## lifetimes  transparancy ###################################################### 
    plt.figure(figsize =(10,6), dpi=200 )
    plt.plot(Frequency_val, lifetime0_values, color='red', linewidth=0.5, alpha=0.5, zorder=2)
    # 2. Create an RGBA array for the markers
    # We define a base color (Red: 1, Green: 0, Blue: 0)
    # Then we use fraction_RMS1 as the Alpha channel (the 4th column)
    alpha_values = np.clip(df_results['fraction_RMS1[m/s]'], None, 1.0)
    rgba_colors = np.zeros((len(df_results['fraction_RMS1[m/s]']), 4))
    rgba_colors[:, 0] = 1.0  # Red channel to max
    rgba_colors[:, 3] = alpha_values # Alpha channel controlled by your data

    # 3. Scatter plot with variable transparency
    # marker='o' for circles
    # facecolors=rgba_colors applies the transparency array
    plt.scatter(Frequency_val, lifetime0_values, 
                facecolors=rgba_colors, 
                edgecolors='none', 
                marker='o', 
                s=15, 
                zorder=4,
                label = 'First Oscillator')

    ######################################################## lifetime2  transparancy ########################## 
    plt.plot(Frequency_val2, lifetime2_values, color='blue', linewidth=0.5, alpha=0.5, zorder=1)
    # 2. Create an RGBA array for the markers
    # We define a base color (Red: 1, Green: 0, Blue: 0)
    # Then we use fraction_RMS1 as the Alpha channel (the 4th column)
    alpha_values = np.clip(df_results['fraction_RMS2[m/s]'], None, 1.0)
    rgba_colors = np.zeros((len(df_results['fraction_RMS2[m/s]']), 4))
    rgba_colors[:, 2] = 1.0  # Red channel to max
    rgba_colors[:, 3] =  alpha_values # Alpha channel controlled by your data

    # 3. Scatter plot with variable transparency
    # marker='o' for circles
    # facecolors=rgba_colors applies the transparency array
    plt.scatter(Frequency_val2, lifetime2_values, 
                facecolors=rgba_colors, 
                edgecolors='none', 
                marker='s', 
                s=20, 
                zorder=3,
                label = 'Second Oscillator')
    plt.xlabel('Frequency of oscillator [1/d]')
    plt.axhline(y=timespan_obs, color='black', linestyle='--', label='Timespan_obs')
    
    plt.legend()
    leg = plt.legend()
    for lh in leg.legend_handles: 
        lh.set_alpha(1) # This forces the legend icons to be 100% opaque
    plt.ylabel('lifetime [d]')
    #plt.title('1D GP Periodogram '+name)
    
    #plt.ylim([0,2*timespan_obs])
    plt.grid(which='major', linestyle='-')
    plt.xlim([0.0,1.0])
    plt.yscale('log')
    plt.minorticks_on()  # This enables minor ticks
    plt.grid(which='minor', color='#DDDDDD', linestyle='--', alpha=0.6)
    plt.savefig("lifetimes "+name+" GP periodogram after gridsearch on initial parameters transparant.png")
    plt.show()    

    ############################## lifetimes zoomed in transparancy ###################################################### 
    plt.figure(figsize =(10,6), dpi=200 )
    plt.plot(Frequency_val, lifetime0_values, color='red', linewidth=0.5, alpha=0.5, zorder=2)
    # 2. Create an RGBA array for the markers
    # We define a base color (Red: 1, Green: 0, Blue: 0)
    # Then we use fraction_RMS1 as the Alpha channel (the 4th column)
    alpha_values = np.clip(df_results['fraction_RMS1[m/s]'], None, 1.0)
    rgba_colors = np.zeros((len(df_results['fraction_RMS1[m/s]']), 4))
    rgba_colors[:, 0] = 1.0  # Red channel to max
    rgba_colors[:, 3] = alpha_values # Alpha channel controlled by your data

    # 3. Scatter plot with variable transparency
    # marker='o' for circles
    # facecolors=rgba_colors applies the transparency array
    plt.scatter(Frequency_val, lifetime0_values, 
                facecolors=rgba_colors, 
                edgecolors='none', 
                marker='o', 
                s=15, 
                zorder=4,
                label = 'First Oscillator')

    ######################################################## lifetime2 zoomed in transparancy ########################## 
    plt.plot(Frequency_val2, lifetime2_values, color='blue', linewidth=0.5, alpha=0.5, zorder=1)
    # 2. Create an RGBA array for the markers
    # We define a base color (Red: 1, Green: 0, Blue: 0)
    # Then we use fraction_RMS1 as the Alpha channel (the 4th column)
    alpha_values = np.clip(df_results['fraction_RMS2[m/s]'], None, 1.0)
    rgba_colors = np.zeros((len(df_results['fraction_RMS2[m/s]']), 4))
    rgba_colors[:, 2] = 1.0  # Red channel to max
    rgba_colors[:, 3] =  alpha_values # Alpha channel controlled by your data

    # 3. Scatter plot with variable transparency
    # marker='o' for circles
    # facecolors=rgba_colors applies the transparency array
    plt.scatter(Frequency_val2, lifetime2_values, 
                facecolors=rgba_colors, 
                edgecolors='none', 
                marker='s', 
                s=20, 
                zorder=3,
                label = 'Second Oscillator')
    plt.xlabel('Frequency of oscillator [1/d]')
    plt.axhline(y=timespan_obs, color='black', linestyle='--', label='Timespan_obs')
    
    plt.legend()
    leg = plt.legend()
    for lh in leg.legend_handles: 
        lh.set_alpha(1) # This forces the legend icons to be 100% opaque
    plt.ylabel('lifetime [d]')
    #plt.title('1D GP Periodogram '+name)
    
    #plt.ylim([0,2*timespan_obs])
    plt.grid(which='major', linestyle='-')
    plt.xlim([0.0,0.1])
    plt.yscale('log')
    plt.minorticks_on()  # This enables minor ticks
    plt.grid(which='minor', color='#DDDDDD', linestyle='--', alpha=0.6)
    plt.savefig("lifetimes "+name+" GP periodogram after gridsearch on initial parameters transparant zoomed in.png")
    plt.show()


    ############################## lifetimes colored spectrum ###################################################### 
    plt.figure(figsize =(10,6), dpi=200 )
    plt.plot(Frequency_val, lifetime0_values, color='gray', linewidth=0.5, alpha=0.5, zorder=2)
    sc = plt.scatter(Frequency_val, lifetime0_values, 
                 c=df_results['fraction_RMS1[m/s]'], 
                 cmap='rainbow', 
                 s=10, 
                 edgecolor='none', 
                 zorder=4,
                 label = 'First Oscillator')
    # 2. Add a colorbar to explain what the colors represent
    cbar = plt.colorbar(sc)
    cbar.set_label('Fraction of RMS explained by first oscillator')

    plt.plot(Frequency_val2, lifetime2_values, color='gray', linewidth=0.5, alpha=0.5, zorder=1)
    sc = plt.scatter(Frequency_val2, lifetime2_values, 
                 c=df_results['fraction_RMS2[m/s]'], 
                 cmap='rainbow', 
                 s=10, 
                 marker = 's',
                 edgecolor='none', 
                 zorder=3,
                 label = 'Second Oscillator')
    plt.xlabel('Frequency of oscillator [1/d]')
    plt.axhline(y=timespan_obs, color='black', linestyle='--', label='Timespan_obs')
    plt.legend()
    plt.ylabel('lifetime [d]')
    #plt.title('1D GP Periodogram '+name)
    
    #plt.ylim([0,2*timespan_obs])
    plt.grid(which='major', linestyle='-')
    #plt.xlim([0.0,0.5])
    plt.yscale('log')
    plt.minorticks_on()  # This enables minor ticks
    plt.grid(which='minor', color='#DDDDDD', linestyle='--', alpha=0.6)
    plt.savefig("lifetimes "+name+" GP periodogram after gridsearch on initial parameters colored spectrum.png")
    plt.show()    


    ##################################lifetime 2 zoomed in colored spectrum
    plt.figure(figsize =(10,6), dpi=200 )
    plt.plot(Frequency_val, lifetime0_values, color='gray', linewidth=0.5, alpha=0.5, zorder=2)
    sc = plt.scatter(Frequency_val, lifetime0_values, 
                 c=df_results['fraction_RMS1[m/s]'], 
                 cmap='rainbow', 
                 s=10, 
                 edgecolor='none', 
                 zorder=4,
                 label = 'First Oscillator')
    # 2. Add a colorbar to explain what the colors represent
    cbar = plt.colorbar(sc)
    cbar.set_label('Fraction of RMS explained by first oscillator')

    plt.plot(Frequency_val2, lifetime2_values, color='gray', linewidth=0.5, alpha=0.5, zorder=1)
    sc = plt.scatter(Frequency_val2, lifetime2_values, 
                 c=df_results['fraction_RMS2[m/s]'], 
                 cmap='rainbow', 
                 s=10, 
                 marker = 's',
                 edgecolor='none', 
                 zorder=3,
                 label = 'Second Oscillator')
    plt.xlabel('Frequency of oscillator [1/d]')
    plt.axhline(y=timespan_obs, color='black', linestyle='--', label='Timespan_obs')
    plt.legend()
    plt.ylabel('lifetime [d]')
    #plt.title('1D GP Periodogram '+name)
    
    #plt.ylim([0,2*timespan_obs])
    plt.grid(which='major', linestyle='-')
    plt.xlim([0.0,0.1])
    plt.yscale('log')
    plt.minorticks_on()  # This enables minor ticks
    plt.grid(which='minor', color='#DDDDDD', linestyle='--', alpha=0.6)
    plt.savefig("lifetimes "+name+" GP periodogram after gridsearch on initial parameters colored spectrum zoomed in.png")
    plt.show() 
#    max_row_list=[]
#    print("type of results = ",type(result))
#    print("type of S0_valurs is = ", type(S0_values))
#    for w0 in w0_list:
#        #print("type of w0 = ",type(w0))
#        max_row = max((row for row in result if row[1] == float(w0)), key=lambda x: x[3])
#        max_row_list.append(max_row)
#    S0_max = [j[0] for j in max_row_list]
#    w0_max = [j[1] for j in max_row_list]
#    Q_max = [j[2] for j in max_row_list]
#    delta_log_like_max = [j[3] for j in max_row_list]
#    Onebylifetime_max = [j[4] for j in max_row_list]
    
    #GLS log likelihood
#    ls = LombScargle(t, y,yerr)
#    power = ls.power(np.array(w0_max)/float(2*np.pi))
#    N=len(t)
#    GLSlogL = -N/2 * np.log(1 - power)
#    GLSdeltaLogL=GLSlogL-null_log_like
#    plt.figure(figsize=(16,9))
#    plt.plot(np.array(w0_max)/float(2*np.pi),delta_log_like_max,'r-',marker='^')
#    #plt.plot(np.array(w0_max)/float(2*np.pi),GLSdeltaLogL,'b-',marker='^',label="GLS")
#    plt.title("GP Periodogram Delta log likelihood GJ 686")
#    plt.xlabel("frequency [1/d]")
#    plt.ylabel("delta log likelihood")
#    plt.legend()
#    plt.savefig("GP periodgoram delta log likelihood GJ 686 grid.png")
#    
#    plt.show()
    
    #Frequency_val_max = np.array(w0_max) / (2 * np.pi)
    #max_index = delta_log_like_max.index(max(delta_log_like_max))
    # Frequency calculations
    #Frequency_val_max = np.array(w0_max) / (2 * np.pi)
    #max_likelihood_freq_max=Frequency_val_max[max_index]
    # Create the figure and subplots
    fig, axs = plt.subplots(5, 1, figsize=(20, 12), sharex=True, dpi=300)

    # First subplot: delta log likelihood
    axs[0].plot(Frequency_val, delta_log_like,'r-',marker='.')
    axs[0].set_ylabel("Delta Log Likelihood")
    #axs[0].set_title(f"1D GP Periodogram {name} first oscillator")
    axs[0].grid()
    #axs[0].set_yscale('log')
    #axs[0].set_ylim([30,60 ])

    # Second subplot: Q values
    axs[1].plot(Frequency_val, Q0_values,'g-',marker='.')
    axs[1].set_ylabel("Q0 ")
    axs[1].grid()
    axs[1].set_yscale('log')
    #axs[1].set_ylim([0, 100])


    # Third subplot: S0 values
    axs[2].plot(Frequency_val, S0_values,'b-',marker='.')
    #axs[2].set_xlabel("Frequency [1/d]")
    axs[2].set_ylabel("S0 [(m/s)^2]")
    axs[2].grid()
    axs[2].set_yscale('log')
    #axs[2].set_ylim([0, 100])

    # Third subplot: S0 values
    axs[3].plot(Frequency_val, lifetime0_values,'m-',marker='.')
    #axs[3].set_xlabel("Frequency [1/d]")
    axs[3].set_ylabel("lifetime0 values [d]")
    axs[3].grid()
    axs[3].set_yscale('log')
    #axs[3].set_ylim([0, 1000])
    
    # fourth subplot: jitter values
    axs[4].plot(Frequency_val, jitter_values,'c-',marker='.')
    axs[4].set_xlabel("Frequency [1/d]")
    axs[4].set_ylabel("jitter values [d]")
    axs[4].grid()
    axs[4].set_yscale('log')
    #axs[3].set_ylim([0, 1000])

    # Adjust layout and save
    #plt.xlim([0.02, 0.07])
    plt.tight_layout()
    plt.savefig(f"GP_periodogram_ {name} _1D first oscillator after gridsearch on initial parameters.png")
    plt.show(block=False)
    plt.close()
    
    fig, axs = plt.subplots(5, 1, figsize=(20, 12), sharex=True, dpi=300)

    # First subplot: delta log likelihood
    axs[0].plot(Frequency_val*2, delta_log_like,'r-',marker='.')
    axs[0].set_ylabel("Delta Log Likelihood")
    #axs[0].set_title(f"1D GP Periodogram {name} second oscillator")
    axs[0].grid()
    #axs[0].set_yscale('log')
    #axs[0].set_ylim([30,60])

    # Second subplot: Q values
    axs[1].plot(Frequency_val*2, Q2_values,'g-',marker='.')
    axs[1].set_ylabel("Q2")
    axs[1].grid()
    axs[1].set_yscale('log')
    #axs[1].set_ylim([0, 100])


    # Third subplot: S0 values
    axs[2].plot(Frequency_val*2, S2_values,'b-',marker='.')
    #axs[2].set_xlabel("Frequency [1/d]")
    axs[2].set_ylabel("S2 [(m/s)^2]")
    axs[2].grid()
    axs[2].set_yscale('log')
    #axs[2].set_ylim([0, 100])

    # Third subplot: S0 values
    axs[3].plot(Frequency_val*2, lifetime2_values,'m-',marker='.')
    #axs[3].set_xlabel("Frequency [1/d]")
    axs[3].set_ylabel("lifetime2 values [d]")
    axs[3].grid()
    axs[3].set_yscale('log')
    #axs[3].set_ylim([0, 1000])
    # fourth subplot: jitter values
    axs[4].plot(Frequency_val*2, jitter_values,'c-',marker='.')
    axs[4].set_xlabel("Frequency [1/d]")
    axs[4].set_ylabel("jitter values [d]")
    axs[4].grid()
    axs[4].set_yscale('log')
    #axs[3].set_ylim([0, 1000])
    
    # Adjust layout and save
    #plt.xlim([0.02, 0.07])
    plt.tight_layout()
    plt.savefig(f"GP_periodogram_{name}_1D_second oscillator after gridsearch on initial parameters.png")
    plt.show(block=False)
    plt.close()


    ############################. Null model.  #####################################
    fig, axs = plt.subplots(5, 1, figsize=(20, 12), sharex=True, dpi=300)

    # First subplot: delta log likelihood
    axs[0].plot(Frequency_val, null_log_like,'r-',marker='.')
    axs[0].set_ylabel("Null Log Likelihood")
    #axs[0].set_title(f"1D GP Periodogram {name} first oscillator")
    axs[0].grid()
    #axs[0].set_yscale('log')
    #axs[0].set_ylim([30,60 ])

    # Second subplot: Q values
    axs[1].plot(Frequency_val, null_Q0_values,'g-',marker='.')
    axs[1].set_ylabel("Q0 ")
    axs[1].grid()
    axs[1].set_yscale('log')
    #axs[1].set_ylim([0, 100])


    # Third subplot: S0 values
    axs[2].plot(Frequency_val, null_S0_values,'b-',marker='.')
    #axs[2].set_xlabel("Frequency [1/d]")
    axs[2].set_ylabel("S0 [(m/s)^2]")
    axs[2].grid()
    axs[2].set_yscale('log')
    #axs[2].set_ylim([0, 100])

    # Third subplot: S0 values
    axs[3].plot(Frequency_val, null_lifetime0_values,'m-',marker='.')
    #axs[3].set_xlabel("Frequency [1/d]")
    axs[3].set_ylabel("lifetime0 values [d]")
    axs[3].grid()
    axs[3].set_yscale('log')
    #axs[3].set_ylim([0, 1000])

    # fourth subplot: jitter values
    axs[4].plot(Frequency_val, null_jitter_values,'c-',marker='.')
    axs[4].set_xlabel("Frequency [1/d]")
    axs[4].set_ylabel("jitter values [d]")
    axs[4].grid()
    axs[4].set_yscale('log')
    #axs[3].set_ylim([0, 1000])
    
    #axs[3].set_ylim([0, 100])
    # Adjust layout and save
    #plt.xlim([0.02, 0.07])
    plt.tight_layout()
    plt.savefig(f"Null_GP_periodogram_ {name} _1D first oscillator after gridsearch on initial parameters.png")
    plt.show(block=False)
    plt.close()
    
    fig, axs = plt.subplots(5, 1, figsize=(20, 12), sharex=True, dpi=300)

    # First subplot: delta log likelihood
    axs[0].plot(Frequency_val*2, null_log_like,'r-',marker='.')
    axs[0].set_ylabel("Null Log Likelihood")
    #axs[0].set_title(f"1D GP Periodogram {name} second oscillator")
    axs[0].grid()
    #axs[0].set_yscale('log')
    #axs[0].set_ylim([30,60])

    # Second subplot: Q values
    axs[1].plot(Frequency_val*2, null_Q2_values,'g-',marker='.')
    axs[1].set_ylabel("Q2")
    axs[1].grid()
    axs[1].set_yscale('log')
    #axs[1].set_ylim([0, 100])


    # Third subplot: S0 values
    axs[2].plot(Frequency_val*2, null_S2_values,'b-',marker='.')
    #axs[2].set_xlabel("Frequency [1/d]")
    axs[2].set_ylabel("S2 [(m/s)^2]")
    axs[2].grid()
    axs[2].set_yscale('log')
    #axs[2].set_ylim([0, 100])

    # Third subplot: S0 values
    axs[3].plot(Frequency_val*2, null_lifetime2_values,'m-',marker='.')
    #axs[3].set_xlabel("Frequency [1/d]")
    axs[3].set_ylabel("lifetime2 values [d]")
    axs[3].grid()
    axs[3].set_yscale('log')
    #axs[3].set_ylim([0, 1000])

    # fourth subplot: jitter values
    axs[4].plot(Frequency_val*2, null_jitter_values,'c-',marker='.')
    axs[4].set_xlabel("Frequency [1/d]")
    axs[4].set_ylabel("jitter values [d]")
    axs[4].grid()
    axs[4].set_yscale('log')
    #axs[3].set_ylim([0, 1000])
    
    # Adjust layout and save
    #plt.xlim([0.02, 0.07])
    plt.tight_layout()
    plt.savefig(f"Null_GP_periodogram_{name}_1D_second oscillator after gridsearch on initial parameters.png")
    plt.show(block=False)
    plt.close()

S0_1, S0_2, Q_1, Q_2, delta_log_like, w0, w_2, jitter, null_S0_1, null_S0_2, null_Q_1, null_Q_2, null_jitter, null_log_like = highest_delta_log_likelihood_solution

# Reconstruct the kernel with optimized parameters
log_Q_lowest = np.log(0.501)
bounds = {"log_S0": (-15, 15), "log_Q": (log_Q_lowest, 15), "log_omega0": (-15, 15)}
term1 = terms.SHOTerm(log_S0=np.log(S0_1), log_Q=np.log(Q_1), log_omega0=np.log(w0), bounds=bounds)
bounds2 = {"log_S0": (-15, 15), "log_Q": (log_Q_lowest, 15), "log_omega0": (-15, 15)}
term2 = terms.SHOTerm(log_S0=np.log(S0_2), log_Q=np.log(Q_2), log_omega0=np.log(w_2), bounds=bounds2)
term1.freeze_parameter("log_omega0")
term2.freeze_parameter("log_omega0")
jitter_term = terms.JitterTerm(log_sigma=np.log(jitter), bounds={"log_sigma": (-15, 15)})
kernel = term1 + term2 + jitter_term

# Set up GP
gp = celerite.GP(kernel, mean=weighted_mean_value)
gp.compute(t, yerr)

# Create dense prediction grid
t_pred = np.linspace(np.min(t), np.max(t), len(t)*10)

# Get predictions (mean and variance) at t_pred
mu, var = gp.predict(y, t_pred, return_var=True)
std = np.sqrt(var)

# Calculate residuals at original t (interpolate mu to t for residuals)
mu_at_t,std_att = gp.predict(y, t, return_var=True)
residuals = y - mu_at_t

# Plot the data, prediction, and residuals using GridSpec
fig = plt.figure(figsize=(10, 6), dpi = 200)
gs = gridspec.GridSpec(3, 1)  # Divide into 3 equal parts vertically; first subplot takes 2 parts

# Top panel: data and prediction (spans rows 0 and 1)
ax1 = fig.add_subplot(gs[0:2, 0])
ax1.errorbar(t, y, yerr=yerr, fmt='.k', capsize=0, label='Data')
ax1.plot(t_pred, mu, 'r-', label='GP Prediction')
ax1.fill_between(t_pred, mu - std, mu + std, color='r', alpha=0.2, label='1σ Uncertainty')
ax1.set_xlabel('Time [d]')
ax1.set_ylabel('RV [m/s]')
ax1.legend()
ax1.set_title('GP Prediction')

# Bottom panel: residuals (spans row 2)
ax2 = fig.add_subplot(gs[2, 0])
ax2.scatter(t, residuals, c='k', s=10)
ax2.axhline(0, color='r', linestyle='--')
ax2.set_xlabel('Time [d]')
ax2.set_ylabel('Residuals [m/s]')
ax2.set_title('Residuals')
plt.tight_layout()
plt.savefig(name+'gp prediction and residual.png')
plt.show()
   
residual_df = pd.DataFrame({
    "t[d]": t,
    "Residual[m/s]": residuals,
    "E_residuals[m/s]": yerr
})

# Save to CSV
residual_df.to_csv(name+" residuals after double SHO gp prediction with jitter.csv", index=False)
