#!/usr/bin/env python3 
import numpy as np
import soundfile as sf
import librosa
import librosa.display
import scipy
from scipy.signal import chirp
from scipy import signal
import matplotlib.pyplot as plt

from expsine import generate_ess

print("Real RIR Test")
rir, fs = sf.read('Audio/h251_Hallway_MITCampus_1txts.wav')
print("Sample rate of the RIR", fs)
# chirp = librosa.chirp(fmin=10, fmax=fs/2, sr=fs, length=fs*10, linear=False) # ESS (exponential sine sweep)
#start_freq, end_freq = 22000, 40000
#duration = 10
#t = np.linspace(0, duration, duration * fs, endpoint=False)
#f_inst = np.linspace(start_freq, end_freq, duration * fs)
#phase = 2 * np.pi * np.cumsum(f_inst) / fs
#chirp = np.sin(phase)

chirp, inv = generate_ess("ess_32k.wav", "inv_ess_32k.wav", T=3.0, f1=10, f2=16000, sample_rate=fs)

fig1, (ax_chirp, ax_inverse_filter, ax_recorded) = plt.subplots(nrows=3, sharex=True, sharey=True)
ax_chirp.plot(chirp)
ax_chirp.set_title('chirp signal (ESS)')
ax_inverse_filter.plot(inv)
ax_inverse_filter.set_title('inverse filter')

fig, ax = plt.subplots()
s_exponential = librosa.stft(y=chirp)
librosa.display.specshow(librosa.amplitude_to_db(s_exponential, ref=np.max), x_axis='time', y_axis='hz', ax=ax)
ax.set(title='chirp signal (ESS) in TF domain')
ax.label_outer()

x = chirp
y = signal.convolve(rir, x)
ax_recorded.plot(y)
ax_recorded.set_title('recorded signal')

def deconvolve(y, x):
	len_y = len(y)
	len_x = len(x)
	len_padded = max(len_y, len_x)
	y_padded = np.pad(y, (0, len_padded-len_y))
	x_padded = np.pad(x, (0, len_padded-len_x))
	Y = scipy.fft.fft(y_padded)
	X = scipy.fft.fft(x_padded)
	ATF = Y / X
	rir = np.real(scipy.fft.ifft(ATF))
	return rir

# rir_recovered = deconvolve(y, x)
ir = signal.convolve(y, inv)
max_value = max(np.amax(ir), -np.amin(ir))
scale = 0.5 / max_value
ir *= scale
ir = ir[np.argmax(ir):]

fig2, (ax_rir, ax_rir_recovered) = plt.subplots(nrows=2, sharex=True, sharey=True)
ax_rir.plot(rir)
ax_rir.set_title('rir')
ax_rir_recovered.plot(ir)
ax_rir_recovered.set_title('recovered rir')

fig3, (ax_rir_tf, ax_rir_recovered_tf) = plt.subplots(nrows=2, sharex=True)
rir_tf = librosa.stft(y=rir)
librosa.display.specshow(librosa.amplitude_to_db(rir_tf, ref=1.0, amin=1e-10, top_db=None), x_axis='time', y_axis='hz', ax=ax_rir_tf)
ax_rir_tf.set(title='RIR in TF domain')
ax_rir_tf.label_outer()
ir_tf = abs(librosa.stft(y=ir))
D = ir_tf.shape[0]
d = int(D*2000/(fs//2))
ir_tf_under_2000hz = ir_tf[:d, :]
# Compute dB without reference
# spec = librosa.amplitude_to_db(ir_tf)	
# Compute dB relative to peak power
#log_spec = librosa.amplitude_to_db(ir_tf, ref=1.0, amin=1e-10, top_db=None)
log_spec = librosa.amplitude_to_db(ir_tf_under_2000hz, ref=np.max)
i = np.argmin(np.mean(log_spec, axis=0))
ir_tf = ir_tf[:, :i]
log_spec = librosa.amplitude_to_db(ir_tf, ref=1.0, amin=1e-10, top_db=None)

img = librosa.display.specshow(log_spec, x_axis='time', y_axis='hz', ax=ax_rir_recovered_tf)
ax_rir_recovered_tf.set(title='Recovered RIR in TF domain')
ax_rir_recovered_tf.label_outer()
#fig3.colorbar(img, ax=ax_rir_recovered_tf, format="%+2.0f dB")

sf.write('chirp_signal.wav', chirp, fs, format='WAV')
sf.write('recorded_signal.wav', y, fs, format='WAV')
sf.write('rir.wav', rir, fs, format='WAV')
sf.write('recovered_rir.wav', ir, fs, format='WAV')
plt.pause(0)
