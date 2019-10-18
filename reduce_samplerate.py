import samplerate
from scipy.io import wavfile
import numpy as np


SECONDS = 3
DOWN_SAMPLING_FACTOR = 5

INT16_MAX = 2.0 ** (16 - 1) + 1

def load_reduced_wav(path):
    rate, wav = wavfile.read(path)  # loads as int16
    wav = (wav[:SECONDS*rate] / INT16_MAX).astype(np.float)  # take first SECONDS and convert to [-1.0, 1.0]
    wav_downsampled = samplerate.resample(wav, 1 / DOWN_SAMPLING_FACTOR)
    return wav_downsampled


if __name__ == '__main__':
    import simpleaudio as sa

    wav_path = '/home/edvard/SharedWithVirtualBox/genres/blues/blues.00000.wav'

    wav_downsampled = load_reduced_wav(wav_path)
    wav = samplerate.resample(wav_downsampled, DOWN_SAMPLING_FACTOR)
    wav = (wav * INT16_MAX).astype(np.int16)

    print('original size: {}'.format(wav.shape))
    print('downsampled size: {}'.format(wav_downsampled.shape))

    play_obj = sa.play_buffer(wav, 1, 2, 22050)  # rate returned by wavfile.read
    play_obj.wait_done()


