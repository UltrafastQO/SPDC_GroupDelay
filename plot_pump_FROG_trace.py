import numpy as np
import matplotlib.pyplot as plt

frog_data = np.loadtxt('pump_FROG_trace.txt')

t = frog_data[:,0]
I = frog_data[:,1]
phase = frog_data[:,2]
I_err = frog_data[:,3]
phase_err = frog_data[:,4]


font = {'family' : 'arial',
        'weight' : 'normal',
        'size'   : 14}

plt.rc('font', **font)

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()

ax1.plot(t, I, '-', color='tab:blue')
ax1.fill_between(t, I-1.95*I_err, I+1.95*I_err, color='tab:blue', alpha=0.25)

ax2.plot(t, phase, '-', color='tab:orange')
ax2.fill_between(t, phase-1.95*phase_err, phase+1.95*phase_err, color='tab:orange', alpha=0.25)

ax1.set_xlabel('Time [fs]')
ax1.set_ylabel('Intensity', color='tab:blue')
ax2.set_ylabel('Phase [rad]', color='tab:orange')

plt.show()
