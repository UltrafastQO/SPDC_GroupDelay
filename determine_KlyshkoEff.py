import numpy as np
import matplotlib.pyplot as plt

from functions_loadData import loadData
from functions_processData import extractCoincSingleRatio
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

coincSig_signal, coincSig_signal_err, coincSig_idler, coincSig_idler_err = extractCoincSingleRatio(dataContainer) # 

# Determine klyshko eff by fitting coinc-singles ratio at 
# low powers to a linear slop to find y-intercept 

def linearFit(x, a, b):
    
    return a*x + b

popt, pcov = curve_fit(linearFit, pumpPowers[20:-1], coincSig_signal[20:-1], p0=[1,1], sigma=coincSig_signal_err[20:-1])
klyshko_signal, klyshko_signal_err = popt[1], np.sqrt(pcov[1,1])*1.95 # 95% confidence interval

popt, pcov = curve_fit(linearFit, pumpPowers[20:-1], coincSig_idler[20:-1], p0=[1,1], sigma=coincSig_idler_err[20:-1])
klyshko_idler, klyshko_idler_err = popt[1], np.sqrt(pcov[1,1])*1.95 # 95% confidence interval

print('Klyshko eff for signal (late) mode is: {} +/- {}'.format(klyshko_signal, klyshko_signal_err))
print('Klyshko eff for idler (early) mode is: {} +/- {}'.format(klyshko_idler, klyshko_idler_err))