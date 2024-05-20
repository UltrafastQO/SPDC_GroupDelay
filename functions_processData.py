import numpy as np
from scipy.ndimage import rotate
from scipy.signal import find_peaks
from lmfit import minimize, Parameters
from scipy.optimize import curve_fit

from thewalrus._torontonian import threshold_detection_prob
import strawberryfields as sf
from strawberryfields.ops import *

def extractJSIs(dataContainer, HWP2_angle=2.0):
    
    JSIs = {}
    
    for dataPoint in dataContainer:
        
        if dataPoint.dataType in ['AE_BL', 'BE_BL', 'AE_AL', 'BE_AL']:
            
            if dataPoint.HWP2_angle == HWP2_angle:
                
                try:
                    JSIs[dataPoint.HWP1_angle] +=  dataPoint.data
                except:
                    JSIs[dataPoint.HWP1_angle] = dataPoint.data
    
    HWP1_angles_list = list(JSIs.keys())
    JSIs_list = list(JSIs.values())
    
    return HWP1_angles_list, JSIs_list
                

def extractGroupDelays(JSIs):
    
    def JSI_model(x, a, b, x0, sigma):
        
        return a*(np.exp(-(x-x0)**2/(2*sigma**2)) + np.exp(-(x+x0)**2/(2*sigma**2))) + b*np.exp(-x**2/(2*sigma**2))
    
    def JSI_residuals(params, x, data):
        
        a =  params['a']
        b = params['b']
        x0 = params['x0']
        sigma = params['sigma']
        
        modelOut = JSI_model(x, a, b, x0, sigma)
        
        return (data-modelOut)    
    
    
    # converting marginal x array to one with wavelength units
    bs = 200 # size of bins [ps/bin]
    GDD = -1032.6 # [ps/nm]
    conversionFactor = bs/GDD    
    
    Num_zeroPad = 500
    nbins = np.shape(JSIs[0])[0]
    
    numJSIs = len(JSIs)
    betas = np.zeros(numJSIs)
    betas_err = np.zeros(numJSIs)
    
    for i in range(numJSIs):
        JSI = JSIs[i]
        
        # rotate by 45 deg
        antiDiagMarg = np.sum(rotate(JSI, angle=45, reshape=False), axis=1)
        
        ## Because I set reshape to False, the function above contracts the
        ## anti-diagonal line to fit nbins. This re-scales the freq bins by
        ## a factor of sqrt(2) too big... Another way to look at this is that 
        ## this procedure finds the "diagonal" separation in the 2D FFT but with
        ## the wrong scaling of the x-axis.
        
        ## Alternative method would be to take 2D FFT of JSI and simply analyze
        ## the marginal (i.e. sum over one axis without using any rotate functions)
        ## However this "throws away" some data and has slightly lower SNR
        
        
        antiDiagMarg_FFT = abs(np.fft.fftshift(np.fft.fft(np.pad(antiDiagMarg, Num_zeroPad, 'constant'))))
                
    
        
        x = np.linspace(0, nbins-1, nbins)
        xLambda = x*conversionFactor
        xLambda = xLambda - xLambda[nbins//2]  # convert to wavelength [nm] with 0 at center 
        xFreq = xLambda*1e-9*3e8/(1558e-9**2) # convert to freq [Hz] with 0 at center 
        
        xFreq = xFreq * np.sqrt(2) # factor to correct for stretching caused by imrotate
        
        df = abs(xFreq[1]-xFreq[0])
        
        maxTime = 1/df
        
        xTime = np.linspace(-maxTime/2, maxTime/2, nbins)
        
        xFreqPadded = np.linspace(-xFreq[0], xFreq[0], 2*Num_zeroPad+nbins)
        xTimePadded = np.linspace(-maxTime/2, maxTime/2, 2*Num_zeroPad+nbins)

        
        centerPeak_idx = (2*Num_zeroPad+nbins) // 2
        
        sidePeak_idx = np.argmax(antiDiagMarg_FFT[centerPeak_idx+40:]) + centerPeak_idx+40
        # look for sideband peak (i.e. away from center peak...)   
        
        params = Parameters()
        
        params.add('a', value=antiDiagMarg_FFT[sidePeak_idx])    
        params.add('b', value=np.max(antiDiagMarg_FFT))
        params.add('x0', value=xTimePadded[sidePeak_idx])
        params.add('sigma', value=1.8e-13)
        
        fitOut = minimize(JSI_residuals, params, args=(xTimePadded, abs(antiDiagMarg_FFT)), method='leastsq')
        popt = np.array(list(fitOut.params.valuesdict().values()))
        pcov = fitOut.covar    
    
    
        betas[i] = popt[2]*1e12 # [ps]
        betas_err[i] = np.sqrt(pcov[2,2]) * 1e12 # [ps]    
    
    return betas, betas_err


def extractParametricGains(dataContainer):
    
    def photonStatsSF(r, eta_E, eta_L):
                    
        sf.hbar = 2
        prog = sf.Program(4)
        eng = sf.Engine("gaussian")
            
        
        with prog.context as q:
            S2gate(r, 0)            | (q[0], q[2])
            LossChannel(eta_E)      | q[0]
            LossChannel(eta_L)      | q[2]
            
            BSgate()                | (q[0], q[1])
            BSgate()                | (q[2], q[3])
            
        
        results = eng.run(prog)
        state = results.state
        
        cov_mat = state.cov()
        mu_vec = state.means()
 
        p = np.zeros((2,2,2,2))
 
        
        # compute all click possibilities using Nico/Jake's torontonian code
        # from thewalrus...
 
        for i in range(0,2):
            for j in range(0,2):
                for k in range(0,2):
                    for l in range(0,2):
                        p[i,j,k,l] = abs(threshold_detection_prob(mu_vec, cov_mat, [i,j,k,l]))
 
                        
        # put click possibilities in the same format as my matrix "click"
        
        clicks_th = np.zeros((4,6))
        
        clicks_th[0,0] = np.sum(p[1,:,:,:]) # AE
        clicks_th[0,1] = np.sum(p[:,1,:,:]) # BE
        clicks_th[0,2] = np.sum(p[:,:,1,:]) # AL
        clicks_th[0,3] = np.sum(p[:,:,:,1]) # BL
        
        clicks_th[1,0] = np.sum(p[1,:,1,:]) # AE_AL
        clicks_th[1,1] = np.sum(p[1,:,:,1]) # AE_BL
        clicks_th[1,2] = np.sum(p[:,1,1,:]) # BE_AL
        clicks_th[1,3] = np.sum(p[:,1,:,1]) # BE_BL
        clicks_th[1,4] = np.sum(p[1,1,:,:]) # AE_BE
        clicks_th[1,5] = np.sum(p[:,:,1,1]) # AL_BL 
        
        clicks_th[2,0] = np.sum(p[1,1,1,:]) # AE_AL_BE
        clicks_th[2,1] = np.sum(p[1,:,1,1]) # AE_AL_BL
        clicks_th[2,2] = np.sum(p[1,1,:,1]) # AE_BE_BL
        clicks_th[2,3] = np.sum(p[:,1,1,1]) # AL_BE_BL
        
        clicks_th[3,0] = p[1,1,1,1] # AE_AL_BE_BL

        
        return clicks_th
    
    def residuals(params, clicks_exp, blank):
        
        # was having issues when I didn't include the dummy "blank"...
        
        eta_E =  params['eta_E']
        eta_L = params['eta_L']
        r = params['r']
        
        clicks_th = photonStatsSF(r, eta_E, eta_L)
        
        # assume equal weight for singles and coincidences in residuals...
        
        return abs(clicks_th-clicks_exp)
    
    
    clicks = {}
    
    for dataPoint in dataContainer:
        
        if dataPoint.dataType in ['clicks']:
            
            if dataPoint.HWP2_angle == 24.5: # only keep if PBS data
                    clicks[dataPoint.HWP1_angle] = dataPoint.data
    
    HWP1_angles_list = list(clicks.keys())
    clicks_list = list(clicks.values())
    
    ## Do fitting
    
    Rs = np.zeros(len(clicks_list))
    Rs_err = np.zeros(len(clicks_list))
    
    for i in range(len(clicks_list)):
        
        click_exp = clicks_list[i][1:, :]/clicks_list[i][0,0] # normalize by num triggs
        
        params = Parameters()
        
        params.add('r', value=1, min=0.01,max=5, vary=True)    
        params.add('eta_E', value=0.069, min=0.005,max=0.99, vary=False)
        params.add('eta_L', value=0.066, min=0.005,max=0.99, vary=False)        
    
        fitOut = minimize(residuals, params, args=(click_exp, None), method='leastsq')
        popt = np.array(list(fitOut.params.valuesdict().values()))
        pcov = fitOut.covar
        
        Rs[i] = popt[0]
        Rs_err[i] = np.sqrt(pcov[0,0])
        
    return Rs, Rs_err

def extractCoincSingleRatio(dataContainer):
    
    clicks = {}
    
    for dataPoint in dataContainer:
        
        if dataPoint.dataType in ['clicks']:
            
            if dataPoint.HWP2_angle == 24.5: # only keep if PBS data
                    clicks[dataPoint.HWP1_angle] = dataPoint.data
    
    HWP1_angles_list = list(clicks.keys())
    clicks_list = list(clicks.values())
    
    N = len(clicks_list)
    
    coinc = np.zeros(N)
    singlesEarly = np.zeros(N)
    singlesLate = np.zeros(N)
    
    for i in range(N):
        
        coinc[i] = np.sum(clicks_list[i][2, 0:4])
        singlesEarly[i] = clicks_list[i][1, 0] + clicks_list[i][1, 1]
        singlesLate[i] = clicks_list[i][1, 2] + clicks_list[i][1, 3]

    
    coincSig_signal = coinc/singlesLate     
    coincSig_idler = coinc/singlesEarly
    
    # propagate error assuming error is sqrt(#counts)
    coincSig_signal_err = coincSig_signal*np.sqrt( (1/coinc) + (1/singlesLate))
    coincSig_idler_err = coincSig_idler*np.sqrt( (1/coinc) + (1/singlesEarly))
    
    return coincSig_signal, coincSig_signal_err, coincSig_idler, coincSig_idler_err



def extractPumpSpectra(dataContainer):
    
    def gauss(x, a, x0, sigma, bg):
    
        return a*np.exp(-(x-x0)**2 / (2*sigma**2)) + bg    
    
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
    std_devs = np.zeros(N)
    std_devs_err = np.zeros(N)
    
    for k in range(N):
        
        wavelengths[:,k] = spectra_list[k][:,0]
        spectrum[:,k] = spectra_list[k][:,1]
        
        # do fitting
        
        popt, pcov = curve_fit(gauss, wavelengths[:,k], spectrum[:,k], p0=[np.max(spectrum[:,k]), 780, 5, 320])
    
        std_devs[k] = popt[2]
        std_devs_err[k] = np.sqrt(pcov[2,2])
    
    return wavelengths, spectrum, std_devs, std_devs_err