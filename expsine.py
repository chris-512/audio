# Tools for computing audio impulse responses using exponential sine sweeps.
# by Thatcher Ulrich
#
# The authors of this script have dedicated the code to the public
# domain. Anyone is free to copy, modify, publish, use, compile, sell,
# or distribute the original tu-testbed code, either in source code form
# or as a compiled binary, for any purpose, commercial or
# non-commercial, and by any means.
# 
# # Exponential Sine Sweep:
# x(t) = sin(K*(exp(t/L) - 1))
#
# T is length of sweep, in seconds
# w1 is starting frequency in radians/sec
# w2 is ending frequency in radians/sec
# L = T / ln(w2/w1)
# K = w1 / L
# x(t) = sin(K * (exp(t/L) - 1))
# 
# The convolutional inverse of the ESS the time-reversal of x(t),
# with amplitude decaying -3dB/octave from high to low frequency,
# because x(t) spectrum is not white (spends more time, and
# therefore energy, on lower frequencies).
# 
# The inverse is:
# f(t) = x(T - t) * w1 / (w1 * exp(t/L))

# References:
#
# Original paper here:
# A. Farina, "Simultaneous measurement of impulse response and distortion with a swept sine
# technique," presented at the 108th AES Convention, Paris, France, February 2000.
# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.33.1614&rep=rep1&type=pdf
#
# Good theory & advanced stuff here:
# "Advancements in impulse response measurements by sine sweeps", Angelo Farina, AES 2007
# http://pcfarina.eng.unipr.it/Public/Papers/226-AES122.pdf
#
# Good practical overview here:
# "SURROUND SOUND IMPULSE RESPONSE: Measurement with the Exponential Sine Sweep;
#  Application in Convolution Reverb", Madeline Carson, Hudson Giesbrecht, Tim Perry.
# http://web.uvic.ca/~timperry/ELEC499SurroundSoundImpulseResponse/Elec499-SurroundSoundIR-PreREVA.pdf
#
# More here:
# "Swept Sine Chirps for Measuring Impulse Response", Ian H. Chan.
# http://www.thinksrs.com/downloads/PDFs/ApplicationNotes/SR1_SweptSine.pdf
#
# The only thing I added to the basic theory is fade-in and fade-out of the sweep, to inhibit clicks
# mentioned by Farina.
#!/usr/bin/env python3
import math
import wave
import numpy as np

def gen_ess(T, f1, f2, sample_rate):
    """Generate an Exponential Sine Sweep signal and its inverse.

    T is time in seconds
    f1 is starting frequency in Hz
    f2 is ending frequency in Hz
    """
    samples = int(sample_rate * T + 0.5)
    # Avoid low or hi-freq energy at the begin/end of the filter by fading for up to 50ms.
    fadesamples = min(sample_rate * 0.050, samples * 0.05)
    twopi = math.pi * 2
    w1 = f1 * twopi
    w2 = f2 * twopi
    L = T / math.log(w2 / w1)
    K = w1 * L
    x = np.zeros(samples)
    A = 0.8  # amplitude
    for i in range(samples):
        t = i / float(sample_rate)
        fadeout = min(1.0, ((samples - i) / float(fadesamples)))
        fadein = min(1.0, i / float(fadesamples))
        fadefactor = min(1.0, i / float(fadesamples), (samples - 1 - i) / float(fadesamples))
        x[i] = A * fadefactor * fadefactor * math.sin(K * (math.exp(t / L) - 1))
    
    f = np.zeros(samples)
    for i in range(samples):
        t = i / float(sample_rate)
        f[i] = x[samples - 1 - i] * w1 / (w1 * math.exp(t/L))

    return (x, f)


def write_ess(ess_filename, inv_filename, T, f1, f2, sample_rate):
    x, f = gen_ess(T, f1, f2, sample_rate)
    writewav(ess_filename, x, x, sample_rate)
    writewav(inv_filename, f, f, sample_rate)


def compute_ir(sweep_filename, inv_filename, ir_filename):
    sl, sr, rate = readwav(sweep_filename)
    f, _, rate2 = readwav(inv_filename)
    if rate != rate2:
        raise ("sweep recording sample rate %s must match inverse sweep signal "
               "sample rate %s!" % (rate, rate2))
    print("Convolving left, please be patient!")
    irl = np.convolve(sl, f)
    print("Now convolving right...")
    irr = np.convolve(sr, f)
    max_value = max(np.amax(irl), -np.amin(irl), np.amax(irr), -np.amin(irr))
    scale = 0.5 / max_value
    irl *= scale
    irr *= scale
    writewav(ir_filename, irl, irr, rate)


def readwav(filename):
    """Read stereo 24-bit .wav file, return tuple of left and right channel data and sample_rate.
    """
    f = wave.open(filename, "r")
    samples = f.getnframes()
    sample_rate = f.getframerate()
    assert f.getnchannels() == 2
    assert f.getsampwidth() == 3

    al = np.zeros(samples)
    ar = np.zeros(samples)
    
    data = f.readframes(samples)
    for i in range(samples):
       offset = i * 3 * 2
       #import pdb; pdb.set_trace()
       #sampl = ord(data[offset + 0]) + 256 * (ord(data[offset + 1]) + 256 * ord(data[offset + 2]))
       sampl = data[offset + 0] + 256 * (data[offset + 1] + 256 * data[offset + 2])
       sampr = data[offset + 3] + 256 * (data[offset + 4] + 256 * data[offset + 5])
       # handle sign
       if sampl >= (1 << 23):
           sampl -= (1 << 24)
       if sampr >= (1 << 23):
           sampr -= (1 << 24)
       sampl = sampl / float(1 << 23)
       sampr = sampr / float(1 << 23)
       al[i] = sampl
       ar[i] = sampr
    f.close()
    return (al, ar, sample_rate)


def writewav(filename, al, ar, sample_rate):
    """Write stereo 24-bit .wav file.
    """
    assert al.shape[0] == ar.shape[0]
    f = wave.open(filename, "w")
    data = bytearray()
    for i in range(al.shape[0]):
        sampl = max(-1.0, min(al[i], 1.0))
        sampl = int(sampl * (1 << 23) + 0.5)
        if sampl < 0:
            sampl += (1 << 24)
        b2 = sampl / (1 << 16)
        b1 = (sampl - b2 * (1 << 16)) / 256
        b0 = sampl - b2 * (1 << 16) - b1 * 256
        if b2 >= 256:
            print("b2: ", b2, " sampl: ", sampl)
        data.append(int(b0))
        data.append(int(b1))
        data.append(int(b2))

        sampr = max(-1.0, min(ar[i], 1.0))
        sampr = int(sampr * (1 << 23) + 0.5)
        if sampr < 0:
            sampr += (1 << 24)
        b2 = sampr / (1 << 16)
        b1 = (sampr - b2 * (1 << 16)) / 256
        b0 = sampr - b2 * (1 << 16) - b1 * 256
        data.append(int(b0))
        data.append(int(b1))
        data.append(int(b2))
    f.setnchannels(2)
    f.setsampwidth(3)
    f.setframerate(sample_rate)
    f.setnframes(al.shape[0])
    f.writeframesraw(data)
    f.close()


if __name__ == "__main__":

    write_ess("ess_44k.wav", "inv_ess_44k.wav", 4.0, 10, 20000, 44100)
    compute_ir("recorded.wav", "inv_ess_44k.wav", "ir_44k.wav")
