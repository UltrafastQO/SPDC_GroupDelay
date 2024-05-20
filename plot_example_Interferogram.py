import numpy as np
import matplotlib.pyplot as plt

from functions_loadData import loadData
from functions_processData import extractJSIs

import seaborn as sns


### Load measured JSIs ###

dataContainer = loadData('data.zip') # load data

HWP1angles, JSIs = extractJSIs(dataContainer, HWP2_angle=2.0)

# converting marginal x array to one with wavelength units
bs = 200 # size of bins[ps]
GDD = -1032.6 # [ps/nm]
conversionFactor = bs/GDD    

Num_zeroPad = 500
nbins = np.shape(JSIs[0])[0]
x = np.linspace(0, nbins-1, nbins)
xLambda = x*conversionFactor
xLambda = xLambda - xLambda[nbins//2]  # convert to wavelength [nm] with 0 at center   
xLambda = xLambda + 1558 # 1558 nm center wavelength

xFreq = xLambda*1e-9*3e8/(1558e-9**2) # convert to freq [Hz] with 0 at center 
df = abs(xFreq[1]-xFreq[0])
maxTime = 1/df
xTime = np.linspace(-maxTime/2, maxTime/2, nbins)
xFreqPadded = np.linspace(-xFreq[0], xFreq[0], 2*Num_zeroPad+nbins)
xTimePadded = np.linspace(-maxTime/2, maxTime/2, 2*Num_zeroPad+nbins)

### Plotting ###

font = {'family' : 'arial',
        'weight' : 'normal',
        'size'   : 18}

plt.rc('font', **font)


i = 15 # plot the 15th JSI
JSI = JSIs[i]

JSI_fft = abs(np.fft.fftshift(np.fft.fft2(np.pad(JSI, Num_zeroPad, 'constant'))))

sns.color_palette("magma", as_cmap=True)

plt.figure(0)
plt.pcolormesh(xFreq[150:350]/1e12, xFreq[150:350]/1e12, JSI[150:350, 150:350], cmap='magma')
plt.xlabel(r'Early bin frequency $\omega_1$ [THz]')
plt.ylabel(r'Late bin frequency $\omega_2$ [THz]')
plt.axis('square')

plt.colorbar()
plt.tight_layout()


plt.figure(1)
plt.pcolormesh(xTimePadded[600:900]*1e12, xTimePadded[600:900]*1e12, JSI_fft[600:900, 600:900], cmap='magma', vmin=0, vmax=1e6)
plt.xlabel('Early photon time [ps]')
plt.ylabel('Late photon time [ps]')
plt.axis('square')
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.xticks([-2, 0, 2])
plt.yticks([-2, 0, 2])

plt.tight_layout()



plt.show()