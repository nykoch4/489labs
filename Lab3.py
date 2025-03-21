import numpy as np
import matplotlib.pyplot as plt

#simulation parameters
dt = 1e-12             #time step (1 ps)
T_total = 5e-9         #total simulation time (5 ns)
t = np.arange(0, T_total, dt)

#input sine wave parameters
freq_in = 1e9          #1 GHz sine wave
amplitude = 2          #2 V amplitude
vin = amplitude * np.sin(2 * np.pi * freq_in * t)

#sampling clock parameters (10 GHz with 50% duty cycle)
f_sampling = 10e9                     #sampling frequency 10 GHz
T_sampling = 1 / f_sampling           #sampling period (100 ps)
clock = ((t % T_sampling) < (T_sampling / 2)).astype(float)  #1 when sampling is active, 0 when holding

#RC sampling parameters
tau = 10e-12            #time constant (10 ps)
v_out = np.zeros_like(t)
v_cap = 0               #initial capacitor voltage

#simulate the sampling circuit
for i in range(len(t)):
    if clock[i] == 1:
        #when the switch is closed, the capacitor charges toward vin following an RC curve.
        v_cap = v_cap + (vin[i] - v_cap) * (dt / tau)
    #when the switch is open, v_cap holds its value.
    v_out[i] = v_cap

#plot
plt.figure(figsize=(10, 6))
plt.plot(t * 1e9, vin, label='Input Signal (1 GHz sine)')
plt.plot(t * 1e9, v_out, label='Sampler Output (ZOH)', linestyle='--')
plt.xlabel('Time (ns)')
plt.ylabel('Voltage (V)')
plt.title('ZOH Sampling Circuit Output')
plt.legend()
plt.grid(True)
plt.show()
