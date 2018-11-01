# -*- coding: utf-8 -*-
"""
Copyright 2018, Prasad Tengse
This Project is Licensed under MIT License
"""

from __future__ import division
import numpy.matlib 
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas
from scipy.io import wavfile
from scipy import signal
import librosa
import librosa.display
from pathlib import Path
import os
import sys


def normalize(ys, amp=1.0):
    """Normalizes a wave array so the maximum amplitude is +amp or -amp.

    ys: wave array
    amp: max amplitude (pos or neg) in result

    returns: wave array
    """
    high, low = abs(max(ys)), abs(min(ys))
    return amp * ys / max(high, low)


samp_rate, audio = wavfile.read("spirit.wav")

audio_rows, audio_cols = np.shape(audio)
if audio_cols == 2:
    print ('Source is Stereo file.')
    audio_l = audio[:,0]
    audio_r = audio[:,1]
    # Lets set a Flag.
    STEREO = True
elif audio_cols == 1:
    print('Source is Mono')
    sys.exit(1)
else:
    print('Multi channel Audio is not suppoted')

if audio_rows / samp_rate  < 0.5:
    print('Audio is Too short to play with')
    sys.exit(1)
else:
    print ("Audio is {0:f} seconds long.".format(audio_rows/samp_rate))

print("Normalizing Numpy Arrays....")
# Normalize 
if STEREO:
    audio_l_n = normalize(audio_l)
    audio_r_n = normalize(audio_r)
        
    ## FFT Shift
    print("FFT...")
    print("Processing Left channel...")
    audio_l_n_fft = np.fft.fft(audio_l_n)
    print ("Processing Right Channel...")
    audio_r_n_fft = np.fft.fft(audio_r_n)
    print("Normalize again")
    audio_l_n_fft_n = normalize(audio_l_n_fft)
    audio_r_n_fft_n = normalize(audio_l_n_fft)
    print("Create center channel...")
    audio_c_fft = audio_l_n_fft_n + audio_r_n_fft_n
    
    # Process Vector products
    print("Computing Roots")
    roots_0 = np.zeros(len(audio_c_fft))
    roots_1 = np.zeros(len(audio_c_fft))
    for i in range(0, len(audio_l_n_fft_n)):
        c_dot_c = np.dot(audio_c_fft[i], audio_c_fft[i])
        c_dot_l_r = np.dot(audio_c_fft[i], (audio_l_n_fft_n[i] + audio_r_n_fft_n[i]))
        l_dot_r = np.dot(audio_l_n_fft_n[i], audio_r_n_fft_n[i])
        root = np.roots([abs(c_dot_c), abs(c_dot_l_r), abs(l_dot_r)])
        roots_0[i] = root [0]
        roots_1[i] = root [1]
        #print(f"{i} is {roots}")
print ("Computing New Channels...")
audio_out_c_fft = roots_0 * audio_c_fft
audio_out_l_fft = audio_l_n_fft_n - audio_out_c_fft
audio_out_r_fft = audio_r_n_fft_n - audio_out_c_fft

print("Back to Time Domain")
audio_out_c = np.fft.ifft(audio_out_c_fft)
audio_out_l = np.fft.ifft(audio_out_l_fft)
audio_out_r = np.fft.ifft(audio_out_l_fft)

print("Normalize again...")
audio_out_c_n = normalize(audio_out_c)
audio_out_l_n = normalize(audio_out_r)
audio_out_r_n = normalize(audio_out_r)
print("Writing Output....")
wavfile.write(filename="voice.wav", rate=samp_rate, data=abs(audio_out_c_n))
wavfile.write(filename="left-voice.wav", rate=samp_rate, data=abs(audio_out_l_n))
wavfile.write(filename="right-voice.wav", rate=samp_rate, data=abs(audio_out_r_n))























