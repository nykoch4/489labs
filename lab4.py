import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#1a
bits = 12
v_rms = 0.2
v_fs = 1.2
v_fs_rms = v_fs / (2 * np.sqrt(2))

# IdealSNR
snr_adc = 6.02 * bits + 1.76


N = 1024
t = np.linspace(0, 1, N)
signal = v_rms * np.sqrt(2) * np.sin(2 * np.pi * 5 * t)
quantized = np.round(((signal + 0.6) / v_fs) * (2**bits - 1)) / (2**bits - 1) * v_fs - 0.6


plt.figure()
plt.plot(t, signal, label="Input Signal")
plt.plot(t, quantized, label="Quantized Signal", linestyle='--')
plt.title("Ideal 12-bit ADC Quantization")
plt.xlabel("Time")
plt.ylabel("Amplitude (V)")
plt.legend()
plt.grid(True)
plt.show()

#1b
v_sin_rms = v_fs_rms
v_noise_rms = 0.5
snr_input_gauss = 20 * np.log10(v_sin_rms / v_noise_rms)

#1c
v_noise_rms_uniform = 1 / np.sqrt(12)
snr_input_uniform = 20 * np.log10(v_sin_rms / v_noise_rms_uniform)

#Results
print(snr_input_gauss, snr_input_uniform)
print(snr_adc)





dnl = np.array([0, -0.5, 0, +0.5, -1, +0.5, +0.5, 0])


actual_thresholds = [0]
for i in range(len(dnl)-1):
    next_edge = actual_thresholds[-1] + 1 + dnl[i]
    actual_thresholds.append(next_edge)


codes = np.arange(8)


plt.figure(figsize=(8, 5))
plt.step(actual_thresholds, codes, where='post', label='Transfer Function')
plt.xlabel('Analog Input (in LSBs)')
plt.ylabel('ADC Output Code')
plt.title('3-bit ADC Transfer Curve with DNL')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()