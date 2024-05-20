import matplotlib.pyplot as plt
import numpy as np
from numpy import exp, pi, sqrt, cos


def f0(dw):
    
    '''Joint spectral mode for first order Magnus term''' 
    
    return sqrt(sqrt(1/pi)) * sqrt(2*tau) * exp (- 2 * tau**2 * dw**2)

def f1(dw):
    
    '''Joint spectral mode for third order Magnus term''' 
    
    return sqrt(3) * f0(dw) * erfi(sqrt(4/3)*tau*dw)


# Parameters

tau = 1
eta_b = 1
eta_a = -1
eta_ab = (eta_a - eta_b)
# assume mu=0 (uncorrelated photons due to GVM)
mua = abs(sqrt(2)*eta_a)
mub = abs(sqrt(2)*eta_b)


# Grid parameters

Nw = 500 # number of grid points
w_max = 1.5 # max freq (in units of freq stds)
dwa = np.linspace(-w_max,w_max,Nw)
dwb = np.linspace(-w_max,w_max,Nw)
DWA, DWB = np.meshgrid(dwa, dwb)

# Defining joint spectral modes

J1 =  f0(DWA) * f0(DWB) 
J3 =  (f0(DWA)*f0(DWB) - f1(DWA)*f1(DWB)) / 12
K3 =  (f0(DWA)*f1(DWB) - f1(DWA)*f0(DWB)) / (4*sqrt(3))

### Plotting ###

font = {'family' : 'arial',
        'weight' : 'normal',
        'size'   : 14}

plt.rc('font', **font)

fig, axs = plt.subplots(1,2, figsize=(8,4))

axs[0].pcolormesh(dwa, dwb, np.zeros((Nw, Nw)), cmap='gray')
axs[0].pcolormesh(dwa, dwb, np.mod(np.angle(J1)+np.pi/2,2*np.pi), cmap='hsv', alpha=np.abs(J1)/np.max(np.abs(J1)), vmin=0,vmax=2*np.pi, shading='auto' )
axs[0].set_aspect('equal')

axs[1].pcolormesh(dwa, dwb, np.zeros((Nw, Nw)), cmap='gray')
axs[1].pcolormesh(dwa, dwb, np.mod(np.angle(J3-1j*K3)+np.pi/2,2*np.pi), cmap='hsv', alpha=np.abs(J3-1j*K3)/np.max(np.abs(J3-1j*K3)), vmin=0,vmax=2*np.pi, shading='auto' )
axs[1].set_aspect('equal')


low_gain = 0.1*J1 + (0.1)**3 * (J3-1j*K3) # epsilon = 0.1
mid_gain = 1.5*J1 + (1.5)**3 * (J3-1j*K3) # epsilon = 1.5
high_gain = 3*J1 + (3)**3 * (J3-1j*K3) # epsilon = 3

fig, axs = plt.subplots(1,3, figsize=(12,4))
axs[0].pcolormesh(dwa*tau, dwb*tau, np.zeros((Nw, Nw)), cmap='gray')
axs[0].pcolormesh(dwa*tau, dwb*tau, np.mod(np.angle(low_gain)+np.pi/2,2*np.pi), cmap='hsv', alpha=np.abs(low_gain)/np.max(np.abs(low_gain)), vmin=0,vmax=2*np.pi, shading='auto' )
axs[0].set_aspect('equal')

axs[1].pcolormesh(dwa*tau, dwb*tau, np.zeros((Nw, Nw)), cmap='gray')
axs[1].pcolormesh(dwa*tau, dwb*tau, np.mod(np.angle(mid_gain)+np.pi/2,2*np.pi), cmap='hsv', alpha=np.abs(mid_gain)/np.max(np.abs(mid_gain)), vmin=0,vmax=2*np.pi, shading='auto' )
axs[1].set_aspect('equal')

axs[2].pcolormesh(dwa*tau, dwb*tau, np.zeros((Nw, Nw)), cmap='gray')
axs[2].pcolormesh(dwa*tau, dwb*tau, np.mod(np.angle(high_gain)+np.pi/2,2*np.pi), cmap='hsv', alpha=np.abs(high_gain)/np.max(np.abs(high_gain)), vmin=0,vmax=2*np.pi, shading='auto' )
axs[2].set_aspect('equal')


plt.show()

