import numpy as np
import matplotlib.pyplot as plt

from functions_loadData import loadData
from functions_processData import extractParametricGains, extractCoincSingleRatio
from scipy.optimize import curve_fit


pumpPowers = np.array([6.09043338e-02, 5.70571150e-02, 5.33185197e-02, 4.96908536e-02,
       4.61763541e-02, 4.27771888e-02, 3.94954543e-02, 3.63331746e-02,
       3.32923001e-02, 3.03747063e-02, 2.75821926e-02, 2.49164813e-02,
       2.23792167e-02, 1.99719636e-02, 1.76962066e-02, 1.55485332e-02,
       1.35402103e-02, 1.16673506e-02, 9.93110895e-03, 8.33255639e-03,
       6.87267878e-03, 5.55237653e-03, 4.37246396e-03, 3.33366880e-03,
       2.43663173e-03, 1.68190602e-03, 1.06995715e-03, 6.01162559e-04,
       2.75811375e-04, 9.41042656e-05]) # units [W]

### Load experimentally measured group delays ###

print('Loading experimental data...')

dataContainer = loadData('data.zip')

Rs_exp, Rs_exp_err = extractParametricGains(dataContainer) # extract gains

# Fit gain vs power to a sqrt 

def sqrtFit(x, a):
    
    return np.sqrt(a*x)

popt, pcov = curve_fit(sqrtFit, pumpPowers, Rs_exp, sigma=Rs_exp_err, p0=[1])




### Plotting ###

font = {'family' : 'arial',
        'weight' : 'normal',
        'size'   : 16}

plt.rc('font', **font)

plt.errorbar(pumpPowers*1000, Rs_exp, Rs_exp_err*1.95, fmt='.', color='k')
plt.plot(pumpPowers*1000, sqrtFit(pumpPowers, *popt), 'k--')

plt.xlabel('Pump power [mW]')
plt.ylabel(r'Parametric gain $\epsilon$')


plt.show()