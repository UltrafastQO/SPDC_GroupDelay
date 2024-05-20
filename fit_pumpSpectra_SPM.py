import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from lmfit import minimize, Parameters
from scipy.linalg import expm

from functions_loadData import loadData
from functions_processData import extractParametricGains

def fitPumpSpectraToSPM(dataContainer, plot_on=False):
    
    def density(x, Np):
        return Np * np.exp(-((x) ** 2) / (4 * (sig) ** 2))    
    
    #Defining gaussian pump pulse. Note that now this includes Np in definition
    def pump(x, Np):
        return np.sqrt(Np)*np.exp(-((x) ** 2) / (2 * (sig) ** 2)) / np.power(np.pi * (sig ** 2), 1 / 4)
    
    def theorySPM(w, spm, bg):
    
        pumpz = lambda x: expm( 1j * (spm * dw / vp ** 2) * (x-domain[0]) * density(-w +w[:,np.newaxis], Np))@pump(w, Np)
        
       
        z=l/2  #position of interest, chosen et end of region.

        theorySpectrum = abs(pumpz(z)**2)
        theorySpectrum = theorySpectrum / np.max(theorySpectrum)
        theorySpectrum = theorySpectrum + bg
        theorySpectrum = theorySpectrum / np.max(theorySpectrum)
        
        return theorySpectrum

    
    spectra = {}
    
    for dataPoint in dataContainer:
        
        if dataPoint.dataType in ['pumpSpectrum']:
            
            if dataPoint.HWP2_angle == 24.5: # consider PBS data
                    spectra[dataPoint.HWP1_angle] = dataPoint.data
    
    HWP1_angles_list = list(spectra.keys())
    spectra_list = list(spectra.values()) 
    
    N = len(spectra_list)
    wavelengths = np.zeros((len(spectra_list[0][:,0]), N))
    spectrum = np.zeros((len(spectra_list[0][:,1]), N))
    spms = np.zeros(N)
    spms_err = np.zeros(N)
    
    std_dev_lowPower = 2.160814426 #[nm]
    std_dev_lowPower_freq = std_dev_lowPower*1e-9*3e8/(779.2e-9**2)
    
    # plotting
    
    if plot_on:
        
        fig, axes = plt.subplots(5, 6, figsize=(12, 8))
        axes = axes.ravel()[:N]
    
    #Unpoled domain
    l=1.0
    dz=l/10
    domain = np.arange(-l/2,l/2,dz)    
    
    
    for k in range(N):
        
        wavelengths[:,k] = spectra_list[k][:,0]
        spectrum[:,k] = spectra_list[k][:,1]
        
        freq = 3e8/(wavelengths[:,k]*1e-9)
        f0 = 3e8/(779.2e-9)
        
        # convert to a normalized freq axis
        start_idx = np.argmin(abs(freq-f0-nsig*std_dev_lowPower_freq))
        end_idx = np.argmin(abs(freq-f0+nsig*std_dev_lowPower_freq))
        
        w = np.linspace(-nsig, nsig, end_idx-start_idx) # normalized freq array
        # do fitting
        dw = w[2]-w[1]
        
        expSpectrum = spectrum[start_idx:end_idx,k]
        expSpectrum = expSpectrum / np.max(expSpectrum)
        bg_guess = np.min(expSpectrum)                
        #Np_guess = 0.01
        spm_guess = 0.1
        
        Np = Nps[k]
        
        popt, pcov = curve_fit(theorySPM, w, expSpectrum, p0=[spm_guess, bg_guess])
        
        if plot_on:
            
            ax = axes[k]
            ax.plot(w, expSpectrum, 'k.')
            ax.plot(w, theorySPM(w, *popt), '-')
            ax.set_title('{} mW'.format(np.round(pumpPowers_exp[k]*1e3,1)))
    
        spms[k] = popt[0]
        spms_err[k] = np.sqrt(pcov[0,0]) * 1.95
        
    fitted_spm_value = np.mean(spms[:15]) # ignore fits at low pump powers where
    # the fitting procedure is mostly insensitive to gamma spm
    fitted_spm_err = np.std(spms[:15])*1.95 # return 2 sigma error
    
    if plot_on:
        
        fig.supxlabel(r'Relative frequency [$\sigma$]')
        fig.supylabel(r'Power (arb units)')
        
        plt.show()
    
    return fitted_spm_value, fitted_spm_err


def fitPumpSpectraToSPMWithRealUnits(dataContainer, plot_on=False):
    
    def density(w, P):
        ''' P is the peak power in Watts'''
        return P * np.exp(-((w) ** 2) / (2 * (sig_real) ** 2))    
    
    #Defining gaussian pump pulse. Note that now this includes Np in definition
    def pump(w, P):
        return np.sqrt(P)*np.exp(-((w) ** 2) / (4 * (sig_real) ** 2)) / np.power(np.pi * (sig_real ** 2), 1 / 4)
    
    def theorySPM(w, spm, bg):
    
        pumpz = lambda x: expm( 1j * (spm * dw / vp_Real ** 2) * (x-domain[0]) * density(-w +w[:,np.newaxis], P))@pump(w, P)
        z=l/2  #position of interest, chosen et end of region.

        theorySpectrum = abs(pumpz(z)**2)
        theorySpectrum = theorySpectrum / np.max(theorySpectrum)
        theorySpectrum = theorySpectrum + bg
        theorySpectrum = theorySpectrum / np.max(theorySpectrum)
        
        return theorySpectrum

    
    spectra = {}
    
    for dataPoint in dataContainer:
        
        if dataPoint.dataType in ['pumpSpectrum']:
            
            if dataPoint.HWP2_angle == 24.5: # only keep if PBS data
                    spectra[dataPoint.HWP1_angle] = dataPoint.data
    
    HWP1_angles_list = list(spectra.keys())
    spectra_list = list(spectra.values()) 
    
    N = len(spectra_list)
    wavelengths = np.zeros((len(spectra_list[0][:,0]), N))
    spectrum = np.zeros((len(spectra_list[0][:,1]), N))
    spms = np.zeros(N)
    spms_err = np.zeros(N)
    
    
    vp_Real = 3e8/1.8092 #[m/s] 1.8092 is group index of KTP for pump
    
    std_dev_lowPower = 2.160814426 #[nm]
    std_dev_lowPower_freq = std_dev_lowPower*1e-9*3e8/(779.2e-9**2) #[Hz]
    
    # plotting
    
    if plot_on:
        
        fig, axes = plt.subplots(5, 6, figsize=(12, 8))
        axes = axes.ravel()[:N]
    
    #Unpoled domain
    l=1 # length 
    dz=l/10
    domain = np.arange(-l/2,l/2,dz)    
    
    
    for k in range(N):
        
        wavelengths[:,k] = spectra_list[k][:,0]
        spectrum[:,k] = spectra_list[k][:,1]
        
        freq = 3e8/(wavelengths[:,k]*1e-9)
        f0 = 3e8/(779.2e-9)
        
        w = freq * 2 * np.pi
        dw = w[2]-w[1]
        
        w0 = f0 * 2 * np.pi
        sig_real = std_dev_lowPower_freq*2*np.pi
        
        # convert to a normalized freq axis
        start_idx = np.argmin(abs(w-w0-10*sig_real))
        end_idx = np.argmin(abs(w-w0+10*sig_real))        
        
        w = w[start_idx:end_idx] - w0 # normalized freq array
        # do fitting

        
        expSpectrum = spectrum[start_idx:end_idx,k]
        expSpectrum = expSpectrum / np.max(expSpectrum)
        bg_guess = np.min(expSpectrum)                
        #Np_guess = 0.01
        spm_guess = 3.51e-6
        
        P = pumpPowers_exp[k]
        P = P / (133e-15*200e3) # convert to peak power (since exp meas is for an avg power, 200kHz // 133 fs)
        
        popt, pcov = curve_fit(theorySPM, w, expSpectrum, p0=[spm_guess, bg_guess])

        if plot_on:
            
            ax = axes[k]
            ax.plot(w/2/np.pi/1e12, expSpectrum, 'k.')
            ax.plot(w/2/np.pi/1e12, theorySPM(w, *popt), '-')
            ax.set_title('{} mW'.format(np.round(pumpPowers_exp[k]*1e3,1)))
    
        spms[k] = popt[0]
        spms_err[k] = np.sqrt(pcov[0,0]) * 1.95
        
    fitted_spm_value = np.mean(spms[:15]) # ignore fits at low pump powers where
    # the fitting procedure is mostly insensitive to gamma spm
    fitted_spm_err = np.std(spms[:15])*1.95 # return 2 sigma error
    
    gamma_spm = fitted_spm_value / (vp_Real**2 * 6.626e-34*f0)
    gamma_spm_err = fitted_spm_err / (vp_Real**2 * 6.626e-34*f0)
    
    if plot_on:
        
        fig.supxlabel(r'Relative frequency [THz]')
        fig.supylabel(r'Intensity (arb units)')
        
        plt.show()
    
    return gamma_spm, gamma_spm_err


################################################################################
################################################################################
################################################################################

font = {'family' : 'arial',
        'weight' : 'normal',
        'size'   : 12}

plt.rc('font', **font)


dataContainer = loadData('data.zip') # load data

### Nico/Martin's SPM

vp = 0.1
sig = 1

#Frequency values
N = 351  # Number of frequency values: Always choose a number ending in 1. That way you always have a frequency value = 0
nsig = 10
wi = -nsig
wf = nsig
dw = (wf - wi) / (N - 1)
w = np.linspace(wi, wf, N)


pumpPowers_exp = np.array([6.09043338e-02, 5.70571150e-02, 5.33185197e-02, 4.96908536e-02,
       4.61763541e-02, 4.27771888e-02, 3.94954543e-02, 3.63331746e-02,
       3.32923001e-02, 3.03747063e-02, 2.75821926e-02, 2.49164813e-02,
       2.23792167e-02, 1.99719636e-02, 1.76962066e-02, 1.55485332e-02,
       1.35402103e-02, 1.16673506e-02, 9.93110895e-03, 8.33255639e-03,
       6.87267878e-03, 5.55237653e-03, 4.37246396e-03, 3.33366880e-03,
       2.43663173e-03, 1.68190602e-03, 1.06995715e-03, 6.01162559e-04,
       2.75811375e-04, 9.41042656e-05])



## Uncomment below to obtain spm parameter in NeedALight units ##

## first determine Np (pump strength in NeedALight units)

#numPumpEnergies = len(pumpPowers_exp) 
#Nps = np.flip(np.linspace(1e-8, 0.038, numPumpEnergies)) 

#Rs_exp, Rs_exp_err = extractParametricGains(dataContainer)
#Nps = Rs_exp ** 2 / 250 # empiracally found that these Np values gives the right
                        ## gain values in NeedALight

#spm, spm_err =  fitPumpSpectraToSPM(dataContainer, plot_on=True) 

## Uncomment below to obtain spm parameter in physical units [W-1 m-1] ##

gamma_spm, gamma_spm_err = fitPumpSpectraToSPMWithRealUnits(dataContainer, plot_on=True) 
