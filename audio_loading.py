import samplerate
from scipy.io import wavfile
import numpy as np


START_POINT = 10  # start few seconds in to avoid initial silence of some songs
SECONDS = 4
DOWN_SAMPLING_FACTOR = 5

INT16_MAX = 2.0 ** (16 - 1) + 1

def _slice_and_rescale(wav, rate):
    return (wav[START_POINT * rate:(START_POINT + SECONDS) * rate] / INT16_MAX).astype(np.float)

def load_reduced_wav(path):
    rate, wav = wavfile.read(path)  # loads as int16
    # take first SECONDS and convert to [-1.0, 1.0]
    wav = _slice_and_rescale(wav, rate)
    wav_downsampled = samplerate.resample(wav, 1 / DOWN_SAMPLING_FACTOR)
    rate = rate / DOWN_SAMPLING_FACTOR
    return rate, wav_downsampled

def load_wav(path):
    rate, wav = wavfile.read(path)  # loads as int16
    wav = _slice_and_rescale(wav, rate)
    return rate, wav


if __name__ == '__main__':
    import simpleaudio as sa

    wav_path = '/home/edvard/SharedWithVirtualBox/genres/blues/blues.00001.wav'

    wav_downsampled = load_reduced_wav(wav_path)
    wav = samplerate.resample(wav_downsampled, DOWN_SAMPLING_FACTOR)
    wav = (wav * INT16_MAX).astype(np.int16)

    print('original size: {}'.format(wav.shape))
    print('downsampled size: {}'.format(wav_downsampled.shape))

    play_obj = sa.play_buffer(wav, 1, 2, 22050)  # rate returned by wavfile.read
    play_obj.wait_done()


