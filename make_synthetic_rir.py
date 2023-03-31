"""
This example creates a room with reverberation time specified by inverting Sabine's formula.
This results in a reverberation time slightly longer than desired.
The simulation is pure image source method.
The audio sample with the reverb added is saved back to `examples/samples/guitar_16k_reverb.wav`.
"""
import argparse
from aiosignal import Signal

import os 
import random
import matplotlib.pyplot as plt
import math
import numpy as np
import pyroomacoustics as pra
from scipy.io import wavfile
from scipy import signal
import librosa
import librosa.display
import seaborn as sns
import time
from expsine import gen_ess

from pyroomacoustics.directivities import (
    DirectivityPattern,
    DirectionVector,
    CardioidFamily,
)

methods = ["ism", "hybrid", "anechoic"]

perform_srp_phat = True
display_rir = False
visualize = False
np.random.seed(2)

# Locations of the microphone array
offset_size = 0.5
x = 0.02285
y = 0.02285
z = 0

mic_offsets = [
  [-y, -x, -z],
  [y, -x, z],
  [y, x, -z],
  [-y, x, z]
]
num_mics = len(mic_offsets)
mic_offsets = np.array(mic_offsets)
delay = 1.0

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def get_azimuth(deg):
    return (deg + 180) % 360 - 180


def resample_and_load(wav_file, sample_ratio, mono=True, is_noise=False):
    _, tmp_audio = wavfile.read(wav_file)
    audio, _ = librosa.load(wav_file, sr=sample_ratio, mono=mono)
    if is_noise:
        duration = len(audio) / sample_ratio
        start_t = random.randrange(0, int(duration) - 10)
        audio = audio[start_t * sample_ratio : (start_t + 10) * sample_ratio]
    magnitude = max(int(tmp_audio.max()), int(-tmp_audio.min()))
    return (audio * magnitude).astype(np.int16)

def ReadWavFiles(path, fs=16000, is_noise=False):
    audios = []
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            path_to_wav = line.strip()
            if not path_to_wav.endswith(".wav"):
                continue

            if librosa.get_samplerate(path_to_wav) == 16000:
                _, audio = wavfile.read(path_to_wav)
                if is_noise:
                    duration = len(audio) / 16000
                    start_t = random.randrange(0, int(duration) - 10)
                    audio = audio[start_t * 16000 : (start_t + 10) * 16000]
            else:
                audio = resample_and_load(path_to_wav, fs, is_noise=is_noise)
                
            audios.append(audio)

    return audios

def wav_to_log_spectrogram(x, n_fft=256):
    # magnitude
    spec = abs(librosa.stft(y=x, n_fft=n_fft))
    # magnitude to dB representation (log-spectrogram)
    log_spec = librosa.amplitude_to_db(spec, ref=np.max)
    return log_spec

def display_log_spectrogram(x, n_fft=256, title="untitled", is_waveform=True):
  
  if is_waveform:
    log_spec = wav_to_log_spectrogram(x, n_fft=n_fft)
  else:
    log_spec = x

  fig, ax = plt.subplots()  
  img = librosa.display.specshow(log_spec, x_axis='time', y_axis='hz', ax=ax)
  ax.set(title=title)
  ax.label_outer()
  fig.colorbar(img, ax=ax, format="%+2.0f dB")

def reconstruct_ir_given_response(y, inv, n_fft=256, sr=44100, duration=2.0, num_frames=None):
  ir = signal.convolve(y, inv)
  max_value = max(np.amax(ir), -np.amin(ir))
  scale = 0.5 / max_value
  ir *= scale
  ir = ir[np.argmax(ir):]
  ### ir
  
  ir_tf = abs(librosa.stft(y=ir, n_fft=n_fft))
  
  # Compute dB without reference
  # spec = librosa.amplitude_to_db(ir_tf)	
  
  # Compute dB relative to peak power
  log_spec = librosa.amplitude_to_db(ir_tf, ref=np.max)
  
  # Add padding
  n_frames = num_frames
  if log_spec.shape[1] < n_frames:
    padding = n_frames - log_spec.shape[1]
    log_spec = np.concatenate((log_spec, -80*np.ones((log_spec.shape[0], padding))), axis=-1)
  else:
    log_spec = log_spec[:, :n_frames]
  ### spectrogram of ir
  
  return ir, log_spec # (ir, spectrogram of ir)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Simulates and adds reverberation to a dry sound sample. Saves it into `./examples/samples`."
    )
    parser.add_argument(
        "--method",
        "-m",
        choices=methods,
        default=methods[0],
        help="Simulation method to use",
    )
    parser.add_argument(
        "--data_root",
        default="data",
        help="Directory to save data",
    )
    args = parser.parse_args()

    ### Hyperparameters ###
    # i) The desired reverberation time
    rt60s = [x/10 for x in range(3, 11, 2)]  # seconds
    # ii) Room dimensions
    room_dim_ratio = [(1, 1), (2, 1), (1, 2), (3, 1), (1, 3)]
    room_dims = [[6, 6, 3], [8, 8, 3], [10, 10, 3], [12, 12, 3], [14, 14, 3], [16, 16, 3], [20, 20, 3]]  # meters
    randomness = 4
    ### Hyperparameters ###
    
    fs = 44100 # 44.1kHz
    nfft = 512
    duration = 1.0
  
    # generate 0.5 sec of ess signal with sample rate of `fs`
    ess, inv = gen_ess(T=0.5, f1=100, f2=fs//2, sample_rate=fs)

    k = 0
    for rt60 in rt60s:
      for wh_ratio in room_dim_ratio:
        for room_dim in room_dims:
          for random_idx in range(randomness):
            k += 1
            
            room_dim_resized = [wh_ratio[0] * math.sqrt(room_dim[0]*room_dim[1]/(wh_ratio[0]*wh_ratio[1])),
                                wh_ratio[1] * math.sqrt(room_dim[0]*room_dim[1]/(wh_ratio[0]*wh_ratio[1])),
                                room_dim[2]]
                      
            print('rt60: ', rt60)
            print('wh_ratio: ', wh_ratio)
            print('room_dim: ', room_dim_resized)
                      
            # We invert Sabine's formula to obtain the parameters for the ISM simulator
            e_absorption, max_order = pra.inverse_sabine(rt60, room_dim_resized)
            # Use Sabine's formula to find the wall energy absorption and maximum order of the
            # ISM required to achieve a desired reveberation time (RT60, i.e. the time it takes
            # for the RIR to decay by 60db)
            
            robot_locs = np.array([room_dim_resized[0]/2, room_dim_resized[1]/2, 0]) # at the center of the room
            # random robot location
            dx = np.random.uniform(-room_dim_resized[0]/4, room_dim_resized[0]/4)
            dy = np.random.uniform(-room_dim_resized[0]/4, room_dim_resized[0]/4)
            robot_locs += [dx, dy, 0]
            
            mic_locs = robot_locs + mic_offsets
            
            # random direction
            (azi, colat) = np.random.choice(range(-180, 180)), np.random.choice(range(0, 30))
            
            # create random directivity object
            dir_obj = CardioidFamily(
                orientation=DirectionVector(azimuth=azi, colatitude=colat, degrees=True),
                pattern_enum=DirectivityPattern.HYPERCARDIOID,
            )
            
            emitter_offset = np.array([0.5, 0, 0])
            emitter_locs = robot_locs + emitter_offset # emitter at height 0.5m
            
            def list_to_str(a):
              return ','.join([str(i) for i in a])
            
            simul_config_name = f"data-{k}"
            simul_data_path = os.path.join(args.data_root, simul_config_name)
            if not os.path.exists(simul_data_path):
              os.makedirs(simul_data_path)
            
            with open(os.path.join(simul_data_path, "config.txt"), "w+") as f:
              config=f"random={random_idx},rt60={str(rt60)},room_dim={list_to_str(room_dim_resized)},robot={list_to_str(robot_locs)},emitter={list_to_str(emitter_locs)},emit_dir={list_to_str([azi, colat])}"
              f.write(config + "\n")

            # Initialize the room
            if args.method == "ism":
                room = pra.ShoeBox(
                    room_dim_resized, fs=fs, materials=pra.Material(e_absorption), max_order=max_order,
                )
            elif args.method == "hybrid":
                room = pra.ShoeBox(
                    room_dim_resized,
                    fs=fs,
                    materials=pra.Material(e_absorption),
                    max_order=3,
                    ray_tracing=True,
                    air_absorption=True,
                )
            elif args.method == "anechoic":
                room = pra.AnechoicRoom(fs=fs)

            room.add_source(
                emitter_locs, signal=ess, delay=delay, directivity=dir_obj
            )

            mic_locs = np.c_[mic_locs[0], mic_locs[1], mic_locs[2], mic_locs[3]]

            # finally place the array in the room
            room.add_microphone_array(mic_locs)

            # Run the simulation (this will also build the RIR automatically)
            room.simulate()

            # [4, ...]
            room.mic_array.signals = room.mic_array.signals[:, int(fs):int((duration+1)*fs)]
            log_spec = wav_to_log_spectrogram(room.mic_array.signals, n_fft=nfft)
            num_mics, num_bins, num_frames = log_spec.shape
            
            # plot signal at microphone 1
            fig_mic, axs = plt.subplots(nrows=4, sharex=True, sharey=True)
            axs[0].plot(room.mic_array.signals[0,:])
            axs[1].plot(room.mic_array.signals[1,:])
            axs[2].plot(room.mic_array.signals[2,:])
            axs[3].plot(room.mic_array.signals[3,:])
            
            mngr = plt.get_current_fig_manager()
            # to put it into the upper left corner for example:
            geom = mngr.window.geometry()
            x,y,dx,dy = geom.getRect()
            mngr.window.setGeometry(x, y, dx, dy)
            
            # plt.pause(0)
            
            #fig_rir, ax_rir = plt.subplots()
            #ir = room.rir[3][0] # RIR between mic 1 and source 0
            #ax_rir.plot(ir)
            
            #ir = room.rir[0][0] # RIR between mic 0 and source 0
            #display_log_spectrogram(ir, n_fft=nfft, title='Simulated RIR in TF domain')
            
            err = False
            ir_tf_4ch = []
            for i in range(num_mics):
              response = room.mic_array.signals[i]
              max_val = max(np.amax(response), -np.amin(response))
              if max_val < 0.0001:
                err = True
                break
              
              ir, ir_tf = reconstruct_ir_given_response(response, inv, n_fft=nfft, sr=fs, duration=duration, num_frames=num_frames)
              
              assert ir_tf.shape[0] == num_bins and ir_tf.shape[1] == num_frames # when n_fft=256
              
              display_log_spectrogram(ir_tf, title=f'Recovered RIR-{i} in TF domain', is_waveform=False)
              
              print("IR in spectrogram shape: ", ir_tf.shape)
              ir_tf_4ch.append(ir_tf)
            
            if err:
              continue

            ir_tf_4ch = np.concatenate(ir_tf_4ch, axis=0)
            
            room.mic_array.to_wav(
                os.path.join(simul_data_path, f"recorded_response.wav"),
                norm=True,
                bitdepth=np.int16,
            )

            # Save the log-spec of recorded response
            print(f'Saving {os.path.join(simul_data_path, "y.npy")}')
            np.save(os.path.join(simul_data_path, "y.npy"), log_spec)
            print(f'Saving {os.path.join(simul_data_path, "rir.npy")}')
            np.save(os.path.join(simul_data_path, "rir.npy"), ir_tf_4ch)

            mngr = plt.get_current_fig_manager()
            # to put it into the upper left corner for example:
            geom = mngr.window.geometry()
            x,y,dx,dy = geom.getRect()
            mngr.window.setGeometry(x+800, y, dx, dy)
            
            if perform_srp_phat:
                # Perform SRP-PHAT
                X = pra.transform.stft.analysis(
                    room.mic_array.signals.T, nfft, nfft // 2, win=np.hanning(nfft)
                )
                X = np.swapaxes(X, 2, 0)

                # perform DOA estimation
                doa = pra.doa.algorithms["SRP"](mic_locs, fs, nfft)
                doa.locate_sources(X)

                # evaluate result
                print(doa.azimuth_recon.shape)
                if doa.azimuth_recon.shape[0] != 0:
                    print("Source is estimated at:", get_azimuth(math.degrees(doa.azimuth_recon)))
                else:
                    print("Cannot extract the azimuth!")

            # measure the reverberation time
            # rt60 = room.measure_rt60()
            # print("The desired RT60 was {}".format(rt60_tgt))
            # print("The measured RT60 is {}".format(rt60[1, 0]))

            # RIR = Room Impulse Response
            # plot the RIRs
            if display_rir:
                select = None  # plot all RIR
                # select = (2, 0)  # uncomment to only plot the RIR from mic 2 -> src 0
                # select = [(0, 0), (2, 0)]  # only mic 0 -> src 0, mic 2 -> src 0
                fig, axes = room.plot_rir(select=select, kind="ir")  # impulse responses
                fig, axes = room.plot_rir(select=select, kind="tf")  # transfer function
                fig, axes = room.plot_rir(select=select, kind="spec")  # spectrograms

            if visualize:
              plt.tight_layout()
              fig_mic.show()
              plt.waitforbuttonpress(1.5) # this will wait for indefinite time
              plt.close()
              plt.close(fig_mic)