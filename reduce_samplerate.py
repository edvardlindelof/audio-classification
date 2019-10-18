import simpleaudio as sa
import samplerate
import librosa
from scipy.io import wavfile
import numpy as np


SECONDS = 3
DOWN_SAMPLING_FACTOR = 2

wav_path ='/home/edvard/SharedWithVirtualBox/genres/blues/blues.00000.wav'
rate, wav = wavfile.read(wav_path)
print(rate)
wav, rate = librosa.load(wav_path)
#wav = wav[:SECONDS*rate]
print(wav)

wav = librosa.resample(wav, rate, rate / DOWN_SAMPLING_FACTOR)
wav = (wav * (2.0 ** (16-1) + 1)).astype(np.int16)
print(wav.dtype)

rate = int(rate / DOWN_SAMPLING_FACTOR)
play_obj = sa.play_buffer(wav, 1, 2, rate)
play_obj.wait_done()


##wav_downsampled = samplerate.resample(wav, 1 / DOWN_SAMPLING_FACTOR)
#wav_upsampled = samplerate.resample(wav_downsampled, DOWN_SAMPLING_FACTOR)
##wav_upsampled = (wav_upsampled * (2.0 ** (16-1) + 1)).astype(np.int)
#print(wav)
#print(wav_upsampled)
#print(wav_upsampled.shape)
#play_obj = sa.play_buffer(wav_upsampled, 1, 2, rate)
##play_obj = sa.play_buffer(wav, 1, 2, rate)
#play_obj.wait_done()
#
#print(wav[:10])
#wav = wav[:SECONDS*rate]
#wav = wav[::DOWN_SAMPLING_FACTOR]
#wav = np.ascontiguousarray(wav)  # because of low level C stuff
#rate = int(rate / DOWN_SAMPLING_FACTOR)
#print(wav[:10])
#
#print(wav.shape)
#play_obj = sa.play_buffer(wav, 1, 2, rate)
#print(play_obj.is_playing())
#play_obj.wait_done()

