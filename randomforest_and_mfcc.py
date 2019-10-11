import numpy as np
from scipy.io import wavfile
from python_speech_features import mfcc
from sklearn import ensemble, preprocessing

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--light', action='store_true', help='Lighter setting for debugging etc')
parser.add_argument(
    '--dataset', default='/home/edvard/SharedWithVirtualBox/genres',
    help='Path to dir with a subdir of .wav files for each class'
)


if __name__ == '__main__':
    args = parser.parse_args()

    class_dir_paths = [
        os.path.join(args.dataset, n) for n in os.listdir(args.dataset) if os.path.isdir(os.path.join(args.dataset, n))
    ]
    if args.light:
        class_dir_paths = class_dir_paths[:2]

    N_FEATURES = 13  # == default number of mfccs in python_speech_features

    train_features, val_features = [], []
    train_classes, val_classes = [], []
    for class_dir_path in class_dir_paths:
        class_name = class_dir_path.split('/')[-1]
        wav_names = os.listdir(class_dir_path)
        print('featurizing {} files...'.format(class_name))
        if args.light:
            wav_names = wav_names[:6]
        n_files = len(wav_names)
        for i, wav_name in enumerate(wav_names):
            wav_path = os.path.join(class_dir_path, wav_name)
            (rate, wav) = wavfile.read(wav_path)
            one_second_index = rate
            # TODO check if nfft=rate is a sensible way to handle the warning that is given if not set
            feats = mfcc(wav[:one_second_index], rate, winlen=1, numcep=N_FEATURES, nfft=rate)

            if i < n_files / 2:
                train_features.append(feats)
                train_classes.append(class_name)
            else:
                val_features.append(feats)
                val_classes.append(class_name)

    train_features = np.array(train_features).reshape(-1, N_FEATURES)
    val_features = np.array(val_features).reshape(-1, N_FEATURES)
    train_classes = preprocessing.LabelEncoder().fit_transform(train_classes)
    val_classes = preprocessing.LabelEncoder().fit_transform(val_classes)
    #print(wav.shape)
    #print(wav.shape[0] / rate)  # 30 seconds

    rf = ensemble.RandomForestClassifier(n_estimators=100)
    print('fitting random forest...')
    rf.fit(train_features, train_classes)
    val_pred_classes = rf.predict(val_features)
    accuracy = np.mean(val_classes == val_pred_classes)
    print('accuracy: {}'.format(accuracy))
