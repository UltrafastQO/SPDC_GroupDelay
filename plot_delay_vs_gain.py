import numpy as np
import matplotlib.pyplot as plt

from NeedALight.propagator import JSA, SXPM_prop
from scipy.optimize import curve_fit
from tqdm import tqdm # used for timing loops

from functions_loadData import loadData
from functions_processData import extractJSIs, extractGroupDelays, extractParametricGains

def symmetric_v(vp, sig, l, a):
    '''Helper function which returns vs, vi (signal and idler velocities) in a
    group velocity matching regime'''
    
    vi = vp / (1 - 2 * a * vp / (l * sig))
    vs = vp / (1 + 2 * a * vp / (l * sig))
    return vs, vi

def pump(x):
    
    return np.sqrt(Np)*np.exp(-((x) ** 2) / (2 * (sig) ** 2)) / np.power(np.pi * (sig)**2, 1 / 4)

def density(x):
    
    return Np * np.exp(-((x) ** 2) / ( (sig) ** 2))

def fitPoly(x, beta, gamma, zeta, y0):
    
    return beta*x + gamma*x**2 + zeta*x**3 + y0


### Load experimentally measured group delays ###

print('Loading experimental data...')

dataContainer = loadData('data.zip')
HWP1_angles_list, JSIs = extractJSIs(dataContainer, HWP2_angle=2.0) # extract JSIs when interfering sig/idl
delays_exp, delays_exp_err = extractGroupDelays(JSIs) # [ps]
Rs_exp, Rs_exp_err = extractParametricGains(dataContainer) # extract gains

# flip arrays so that first idx is lowest gain
Rs_exp, Rs_exp_err = np.flip(Rs_exp), np.flip(Rs_exp_err) 
delays_exp, delays_exp_err = np.flip(delays_exp), np.flip(delays_exp_err) 

# ignore last two data points for highest pump powers (saturation effects)
Rs_exp = Rs_exp[:-2] 
Rs_exp_err = Rs_exp_err[:-2]
delays_exp = delays_exp[:-2]
delays_exp_err = delays_exp_err[:-2]


### Compute theory group delays using NeedALight ###

## Parameters ##

N = 251  # Number of frequency values: Always choose a number ending in 1. That way you always have a frequency value = 0
vp = 0.1  # pump velocity
l = 1.0  # amplification region length
sig = 1  # pump wave packet spread
a = 1.61 / 1.13  # from symmetric grp vel matching

vs, vi = symmetric_v(vp, sig, l, a)

spm = 0.223     # value of self-phase modulation for pump (see fit_pumpSpectra_SPM.py)
xpmi = spm*2/3  # value of cross-phase modulation for signal
xpms = spm*2    # value of cross-phase modulation for idler

## Grids ##

#Frequency grid
wi = -10 # units of freq stds
wf = 10
dw = (wf - wi) / (N - 1)
w = np.linspace(wi, wf, N)

#Unpoled domain
dz=l/10 # units of crystal length
domain = np.arange(-l/2,l/2,dz)

# gains
num_gains = len(Rs_exp)
Nps = Rs_exp ** 2 / 250 # empiracally found that these Np values gives the right
                        # gain values in NeedALight

delays_th = np.zeros(num_gains)
delays_th_err = np.zeros(num_gains)
numPhotons = np.zeros(num_gains)

# Calculate the group delays
print('Computing theory delays...')
for i in tqdm(range(num_gains)):
    
    Np = Nps[i]
    
    #Generate propagator with S-X-PM
    T2 = SXPM_prop(vs, vi, vp, 1, spm, xpms, xpmi, pump, density, domain, w)
    
    #Generates JSAs as well as other relevant properties
    J, numPhotons[i], K, M, Nums, Numi = JSA(T2, vs, vi, vp, l, w) 
    
    phase2d = np.angle(J)
    phase1d = np.fliplr(phase2d).diagonal() # consider only the anti-diagonal cross-section
    phase1d = np.unwrap(phase1d)
    
    mid_idx = N // 2
    
    popt, pcov = curve_fit(fitPoly, w[mid_idx-10:mid_idx+10], phase1d[mid_idx-10:mid_idx+10], p0=[-0,0,0,-10]) 
    # polynomial fit on center region
    
    delays_th[i] = 2*popt[0] # 2x linear gradient is the group delay
    delays_th_err[i] = 2*np.sqrt(pcov[0,0]) 
    


# Convert NeedALight units into [ps]

pumpSTD_in_ps = 220e-3 / 2.355 # measured 220fs FWHM (intensity) using FROG

# Need to divide by sqrt(2) since NeedALight freq units are in terms of 
# amplitude STDs

delays_th = delays_th * pumpSTD_in_ps / np.sqrt(2)
delays_th_err = delays_th_err * pumpSTD_in_ps / np.sqrt(2)     
    
Rs_th = np.arcsinh(np.sqrt(numPhotons)) 


betaBBO = 0.833 # group delay [ps] due to 2mm of alpha-BBO

beta0_exp = delays_exp[0] - betaBBO
beta0_th = 0.3253 # theory group delay [ps] due to GVM in 2mm of ppKTP.
                  # calculated using ng[idler] = 1.7538 and ng[signal] = 1.8514



### Plotting ###

font = {'family' : 'arial',
        'weight' : 'normal',
        'size'   : 16}

plt.rc('font', **font)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()

ax1.plot(Rs_th, beta0_th*np.ones(len(Rs_th)), 'k--', label=r'$J_1$')
ax1.plot(Rs_th, beta0_th - 24*pumpSTD_in_ps*Rs_th**2/(np.sqrt(3*np.pi)*(12+Rs_th**2)), 'k-.', label=r'$J_1+J_3$')

ax1.plot(Rs_th, delays_th+beta0_th, '-', label='Numeric', color='k')
ax1.fill_between(Rs_th, (delays_th-1.95*delays_th_err)+beta0_th,  (delays_th+1.95*delays_th_err)+beta0_th, alpha=0.25, color='k')


ax1.errorbar(Rs_exp, delays_exp-betaBBO, yerr=delays_exp_err*1.95, xerr=Rs_exp_err*1.95, fmt='.', label='Experiment', color='k')


ax1.set_xlabel(r'Parametric gain $\varepsilon$')
ax1.set_ylabel(r'Group delay $T$ [ps]')

ax1.set_ylim([0,0.35])
ax1.set_xlim([0,3])
ax1.set_yticks([0.3, 0.2, 0.1, 0.0])

def tick_function(r):
    n = np.sinh(r)**2
    return ["%d" % i for i in n]

new_tick_locations = np.array([0, 1, 1.87, 3])
ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks(new_tick_locations)
ax2.set_xticklabels(tick_function(new_tick_locations))
ax2.set_xlabel(r'Photon pairs per pump pulse $\langle n \rangle$')


ax1.legend()
plt.tight_layout()
plt.show()