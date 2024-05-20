import numpy as np
import matplotlib.pyplot as plt

from functions_loadData import loadData
from functions_processData import extractJSIs, extractParametricGains

# get JSIs when signal/idler are interfered at PBS (HWP_angle=2)

dataContainer = loadData('data.zip')
HWP1angles, JSIs = extractJSIs(dataContainer, HWP2_angle=2.0)
Rs_exp, Rs_exp_err = extractParametricGains(dataContainer)


# plot all JSIs

N = len(HWP1angles)

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


num_rows = 4
num_cols = 7

# Create a figure and a grid of subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(9, 9),subplot_kw={'aspect': 'equal'})

# If N is not a perfect square, you may have some empty subplots at the end.
# To handle this, flatten the axes array and remove the empty subplots.
axes = axes.ravel()[:N]

JSIs_f = list(reversed(JSIs))
Rs_f = np.flip(Rs_exp)

for i, ax in enumerate(axes):
    
    if i < 28:
        
        angle = HWP1angles[i]
        JSI = JSIs_f[i]
         
        pcm = ax.pcolormesh(xFreq/1e12, xFreq/1e12, JSI, cmap='jet')
        ax.set_title(r'$\varepsilon$={}'.format(round(Rs_f[i],2)))
        ax.set_xlim([186,198])
        ax.set_ylim([186,198])

    
    
# Adjust the layout to prevent overlapping titles and labels
plt.tight_layout()

fig.supxlabel(r'$\omega_1$ [THz]')
fig.supylabel(r'$\omega_2$ [THz]')

# Show all the subplots simultaneously
plt.show()