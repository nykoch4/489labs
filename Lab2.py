import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram, windows

#generate sin wave
def generate_sin(freq, amp, Fs, N):
    t = np.arange(N) / Fs
    return t, amp * np.sin(2 * np.pi * freq * t)

#add noise for a given SNR
def add_noise(signal, SNR_dB):
    P_signal = np.mean(signal ** 2)
    P_noise = P_signal / (10**(SNR_dB / 10))
    noise = np.random.normal(0, np.sqrt(P_noise), size=signal.shape)
    return signal + noise

#plot PSD
def plot_psd(signal, Fs, title, window=None):
    if window is not None:
        signal = signal * window(len(signal))
    
    f, psd = periodogram(signal, Fs, scaling='density')

    psd[psd <= 0] = 1e-12
    psd_db = 10 * np.log10(psd)

    plt.figure(figsize=(8, 5))
    plt.plot(f, psd_db, label=title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density (dB/Hz)")
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.show()

#quantization func
def quantize(signal, bits):
    levels = 2 ** bits
    max_val = np.max(np.abs(signal))
    quantized_signal = np.round((signal / max_val) * (levels / 2)) * (max_val / (levels / 2))
    return quantized_signal

#SNR calc func
def calculate_snr(signal, noise):
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    return 10 * np.log10(signal_power / noise_power)

#parameters for Part 1
Fs = 5e6  #sampling rate
Fsig = 2e6  #signal frequency
Amp = 1  #amplitude
N = 1024  #num of samples

#generate sin wave and add noise
t, sine_wave = generate_sin(Fsig, Amp, Fs, N)
noisy_signal_gaussian = add_noise(sine_wave, 50)
plot_psd(noisy_signal_gaussian, Fs, "Noisy Signal PSD (50 dB SNR, Gaussian Noise)")

#calculate SNR for noise
noise_gaussian = noisy_signal_gaussian - sine_wave
snr_gaussian = calculate_snr(sine_wave, noise_gaussian)
print(f"SNR with Gaussian noise: {snr_gaussian:.2f} dB")

#apply windows and compute SNR
for win, name in zip([windows.hann, windows.hamming, windows.blackman],
                      ["Hanning", "Hamming", "Blackman"]):
    windowed_signal = sine_wave * win(len(sine_wave))
    noisy_signal_windowed = add_noise(windowed_signal, 50)
    plot_psd(noisy_signal_windowed, Fs, f"PSD with {name} Window (50 dB SNR)", window=win)

    #calculate SNR for windowed signal
    noise_windowed = noisy_signal_windowed - windowed_signal
    snr_windowed = calculate_snr(windowed_signal, noise_windowed)
    print(f"SNR with {name} window: {snr_windowed:.2f} dB")

#parameters for part 2
Fs_quant = 400e6  #sampling rate
f_signal_quant = 200e6  #signal frequency
T_quant = 1 / f_signal_quant  #signal period
bits_list = [6, 12]  #num of bits

#quant and compute SNR for 6 and 12 bit
for periods in [30, 100]:
    t_quant = np.arange(0, periods * T_quant, 1 / Fs_quant)
    sine_wave_quant = np.sin(2 * np.pi * f_signal_quant * t_quant)

    for bits in bits_list:
        quantized_signal = quantize(sine_wave_quant, bits)
        noise_quant = quantized_signal - sine_wave_quant
        snr_quant = calculate_snr(sine_wave_quant, noise_quant)
        
        print(f"Quantization SNR for {bits}-bit, {periods} periods: {snr_quant:.2f} dB")
        
        plot_psd(quantized_signal, Fs_quant, f"{bits}-bit Quantized Signal PSD ({periods} periods)")

#incommensurate sampling freq
Fs_incommensurate = 410e6
t_inc = np.arange(0, 30 * T_quant, 1 / Fs_incommensurate)
sine_wave_inc = np.sin(2 * np.pi * f_signal_quant * t_inc)

quantized_signal_inc = quantize(sine_wave_inc, 6)
noise_inc = quantized_signal_inc - sine_wave_inc
snr_inc = calculate_snr(sine_wave_inc, noise_inc)

print(f"Incommensurate Sampling SNR: {snr_inc:.2f} dB")
plot_psd(quantized_signal_inc, Fs_incommensurate, "Incommensurate Sampling PSD")


#hanning window and repeat 2c
for bits in bits_list:
    windowed_signal = sine_wave_quant * windows.hann(len(sine_wave_quant))
    quantized_signal_windowed = quantize(windowed_signal, bits)
    noise_windowed = quantized_signal_windowed - windowed_signal
    snr_windowed = calculate_snr(windowed_signal, noise_windowed)

    print(f"Quantization SNR with Hanning window ({bits}-bit): {snr_windowed:.2f} dB")
    plot_psd(quantized_signal_windowed, Fs_quant, f"{bits}-bit Quantized Signal PSD (Hanning)", window=windows.hann)

#add noise for 38 dB SNR and repeat 2c and 2d
target_snr_db = 38
signal_power = np.mean(sine_wave_quant ** 2)
noise_power = signal_power / (10 ** (target_snr_db / 10))
noise = np.sqrt(noise_power) * np.random.uniform(-1, 1, size=sine_wave_quant.shape)

sine_wave_noisy = sine_wave_quant + noise

for bits in bits_list:
    quantized_signal_noisy = quantize(sine_wave_noisy, bits)
    noise_noisy = quantized_signal_noisy - sine_wave_noisy
    snr_noisy = calculate_snr(sine_wave_noisy, noise_noisy)

    print(f"Quantization SNR with added noise ({bits}-bit, 38dB SNR): {snr_noisy:.2f} dB")
    plot_psd(quantized_signal_noisy, Fs_quant, f"{bits}-bit Quantized Signal PSD with Noise")

    #with hanning window
    snr_noisy_hanning = calculate_snr(sine_wave_noisy * windows.hann(len(sine_wave_noisy)), 
                                      noise_noisy * windows.hann(len(noise_noisy)))

    print(f"Quantization SNR with added noise + Hanning ({bits}-bit, 38dB SNR): {snr_noisy_hanning:.2f} dB")
    plot_psd(quantized_signal_noisy, Fs_quant, f"{bits}-bit Quantized Signal PSD with Noise (Hanning)", window=windows.hann)