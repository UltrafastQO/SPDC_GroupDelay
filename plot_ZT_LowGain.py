import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm
from matplotlib.colors import LinearSegmentedColormap
from NeedALight.propagator import JSAK, FtS2
from tqdm import tqdm # used for timing loops

def symmetric_v(vp, sig, l, a):
    vi = vp / (1 - 2 * a * vp / (l * sig))
    vs = vp / (1 + 2 * a * vp / (l * sig))
    return vs, vi

# defining gaussian pump pulse in momentum space with z_0=l
def pump(x, scale=1):
    return np.exp(-((x) ** 2) / (2 * ((sig/vp)/scale) ** 2)) / np.power(np.pi * ((sig/vp)/scale)**2, 1 / 4)*np.exp(1j*x*L)


#Defining fourier transforms and spatial range
Basic_ift = lambda z,k,func: np.sum(np.exp(1j*np.tensordot(z,k,axes=0))/np.sqrt(2*np.pi)*(k[1]-k[0])*func,axis=1)
Basic_ft = lambda z,k,func: np.sum(np.exp(-1j*np.tensordot(k,z,axes=0))/np.sqrt(2*np.pi)*(z[1]-z[0])*func,axis=1) #for when you want to input a vector and not a matrix
Basic_ift2D = lambda z,k,func: np.sum(np.sum(np.exp(1j*np.tensordot(z,k-k[:,np.newaxis],axes=0))/(2*np.pi)*((k[1]-k[0]))*func,axis=1),axis=1) #This is to take the FT of Nums and Numi,note the sign difference for the k's


#Parameters
Np = 0.0005375 # pump strength
vp = 0.1  # pump velocity
L = 1.0  # amplification region length
sig = 1  # pump wave packet spread
a = 1.61 / 1.13  # from symmetric grp vel matching


vs, vi = symmetric_v(vp, sig, L, a)

#For numerical FT
nk = 201
k_ft = 200/L
dk = k_ft/nk
k = np.arange(-k_ft/2,k_ft/2,dk)

sc=1 # for non-aperiodic poled crystal
dz = 0.01
z0 = -1 # starting position of the pump pulse
z_list = np.arange(-0.5, 0.5, dz) # amplification region
domain= np.zeros(len(z_list))+1


ws = vs*k
wi = vi*k

zplot=np.arange(-2,2,dz)

#Play around with this at your leisure. Currently chosen such that all pulses completely leave the crystal.
#Also chosen such that pump pulse ends at z=1.

dt=0.2
t=np.arange(0,20+dt,dt) * 1.0

# Pump params

nkp=801 #Number of momentum values for pump FT
kp_ft = 1000/L 
dkp = kp_ft/nkp
kp = np.arange(-kp_ft/2,kp_ft/2,dkp)

#Pump dispersion relation
wp = vp*kp

#Defining pump envelope and dispersion relations
Lambda = np.zeros([len(t),len(z_list)], dtype=np.complex128)
for i in range(len(t)):
    Lambda[i,:] = np.sqrt(Np)*np.sum(np.exp(-1j*wp[:,np.newaxis]*t[i])*np.exp(1j*kp[:,np.newaxis]*(z_list))*pump(kp[:,np.newaxis],scale=1/sc),axis=0)*dkp/np.sqrt(2*np.pi)


LambdaPlot = np.zeros([len(t),len(zplot)], dtype=np.complex128)
for i in range(len(t)):
    LambdaPlot[i,:] = np.sqrt(Np)*np.sum(np.exp(-1j*wp[:,np.newaxis]*t[i])*np.exp(1j*kp[:,np.newaxis]*(zplot))*pump(kp[:,np.newaxis],scale=1/sc),axis=0)*dkp/np.sqrt(2*np.pi)


    
#Initializing
KT = np.identity(2 * len(k), dtype=np.complex128)
S = np.zeros((len(k),len(k)),dtype=np.complex128)
dk = k[1]-k[0]
dt = t[1]-t[0]

zst = np.zeros_like(t)
zit = np.zeros_like(t)
zpt = np.zeros_like(t)
    
ind = 0    
#Constructing the diagonal blocks
Rs = np.diag(-1j*ws)
Ri = np.diag(1j*wi)

signalPulse = np.zeros((len(zplot), len(t)))
idlerPulse = np.zeros((len(zplot), len(t)))
pumpPulse = np.zeros((len(zplot), len(t)))

for i in tqdm(t):
        
    S = 1j*FtS2(domain,Lambda[ind,:],z_list,k)*dk/np.sqrt(2*np.pi)
    Q = np.block([[Rs,S],[np.conjugate(S),Ri]])
    KT = expm(Q*dt)@KT

    JT, NsT, SchmidtT, MT, NumsT, NumiT = JSAK(KT,dk) #T because they are temporary
    
    
    NumsZT = np.real(Basic_ift2D(zplot,k,NumsT))
    NumiZT = np.real(Basic_ift2D(zplot,k,NumiT))
    Pump_ET =np.real_if_close(LambdaPlot[ind,:]*np.conj(LambdaPlot[ind,:]))
    
    signalPulse[:, ind] = NumsZT
    idlerPulse[:, ind] = NumiZT
    pumpPulse[:, ind] = Pump_ET
    
    zst[ind]=zplot[np.argmax(np.abs(NumsZT))]
    zit[ind]=zplot[np.argmax(np.abs(NumiZT))]
    zpt[ind]=zplot[np.argmax(Pump_ET)]
        
    ind=ind+1


####### Make custom colormaps which becomes transparent at low values

# get colormap
ncolors = 256
color_array = plt.get_cmap('Blues')(range(ncolors))

# change alpha values
color_array[:,-1] = np.linspace(0.0,1.0,ncolors) # linearly increasing alpha

# create a colormap object
map_object1 = LinearSegmentedColormap.from_list(name='viridis_alpha',colors=color_array)

# get colormap
color_array = plt.get_cmap('Reds')(range(ncolors))

# change alpha values
color_array[:,-1] = np.linspace(0.0,1.0,ncolors) # linearly increasing alpha

# create a colormap object
map_object2 = LinearSegmentedColormap.from_list(name='plasma_alpha',colors=color_array)


#############################



font = {'family' : 'arial',
        'weight' : 'normal',
        'size'   : 22}

plt.rc('font', **font)

plt.figure(figsize=(5,5))

plt.pcolormesh(t,zplot, signalPulse/np.max(signalPulse), cmap=map_object1)
plt.pcolormesh(t,zplot, idlerPulse/np.max(idlerPulse), cmap=map_object2)

plt.axhline(y=-0.5, color='grey')
plt.axhline(y=+0.5, color='grey')
plt.plot(t, zst, color='k', linestyle='-.')
plt.plot(t, zit, color='k', linestyle='--')
plt.plot(t, zpt, 'k-')

plt.xlabel(r'Time (units of $\tau$)')
plt.ylabel(r'Space (units of $L$)')

plt.tight_layout()

plt.ylim([-1.00, 0.5])
plt.yticks([-1.0, -0.5, 0, 0.5])


plt.show()
