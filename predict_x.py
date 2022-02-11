from tensorflow.keras.models import load_model
from clean import downsample_mono, envelope, check_dir, split_wavs, save_sample
from kapre.time_frequency import STFT, Magnitude, ApplyFilterbank, MagnitudeToDecibel
from sklearn.preprocessing import LabelEncoder
import numpy as np
from glob import glob
import argparse
import os
import pandas as pd
from tqdm import tqdm

#from pydub import AudioSegment

import matplotlib.pyplot as plt
from scipy.io import wavfile
#import argparse
#import os
#from glob import glob
#import numpy as np
#import pandas as pd
from librosa.core import resample, to_mono
#from tqdm import tqdm
import wavio





#if __name__ == '__main__':

    

#prediction starts

def make_prediction(args):

    model = load_model(args.model_fn,
        custom_objects={'STFT':STFT,
                        'Magnitude':Magnitude,
                        'ApplyFilterbank':ApplyFilterbank,
                        'MagnitudeToDecibel':MagnitudeToDecibel})
    wav_paths = glob('{}/**'.format(args.src_dir), recursive=True)
    wav_paths = sorted([x.replace(os.sep, '/') for x in wav_paths if '.wav' in x])
    classes = sorted(os.listdir(args.src_dir))
    classes.remove("X Song")
    labels = [os.path.split(x)[0].split('/')[-1] for x in wav_paths]
    le = LabelEncoder()
    y_true = le.fit_transform(labels)
    results = []

    for z, wav_fn in tqdm(enumerate(wav_paths), total=len(wav_paths)):
        rate, wav = downsample_mono(wav_fn, args.sr)
        mask, env = envelope(wav, rate, threshold=args.threshold)
        clean_wav = wav[mask]
        step = int(args.sr*args.dt)
        batch = []

        for i in range(0, clean_wav.shape[0], step):
            sample = clean_wav[i:i+step]
            sample = sample.reshape(-1, 1)
            if sample.shape[0] < step:
                tmp = np.zeros(shape=(step, 1), dtype=np.float32)
                tmp[:sample.shape[0],:] = sample.flatten().reshape(-1, 1)
                sample = tmp
            batch.append(sample)
        X_batch = np.array(batch, dtype=np.float32)
        y_pred = model.predict(X_batch)
        y_mean = np.mean(y_pred, axis=0)
        y_pred = np.argmax(y_mean)
        #real_class = os.path.dirname(wav_fn).split('/')[-1]
        print('File: {}, Predicted class: {}'.format(wav_fn.replace("Random Song/Testing/X Song/",""), classes[y_pred]))
        results.append(classes[y_pred])
    Count = [results.count("Acoustic_Guitar"),results.count("Drums"),results.count("Electric_Guitar"),results.count("Flute Family"),results.count("Piano"),results.count("Violin Family"),results.count("Voice")]
    prediction = pd.DataFrame(results,columns=["Predictions"]).to_csv('predictions.csv')
    # print(Count)
    # for k in range len(Count):
    #     if Count[k]== 0:
    #         Count.pop(k)
    #         classes.pop(k)
    #         k-=1
    classes = [classes[i] for i,_ in enumerate(Count) if _ != 0]
    Count = np.array(Count)
    plt.pie(Count[Count != 0],labels=classes,autopct='%1.2f%%')
    plt.show()

    #np.save(os.path.join('logs', args.pred_fn), np.array(results))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Cleaning audio data')
    parser.add_argument('--src_root', type=str, default='Random Song/Full Song',
                        help='directory of audio files in total duration')
    parser.add_argument('--dst_root', type=str, default='Random Song/Testing',
                        help='directory to put audio files split by delta_time')
    parser.add_argument('--delta_time', '-dt', type=float, default=3.0,
                        help='time in seconds to sample audio')
    parser.add_argument('--sr', type=int, default=16000,
                        help='rate to downsample audio')

    parser.add_argument('--fn', type=str, default='3a3d0279',#236cbab1,9a4bfb69,81d9f077,19105b8a,103f29e5,3707da71,  3a3d0279  ,37cebbd4,1e3391d5,642aba58
                        help='file to plot over time to check magnitude')
    parser.add_argument('--threshold', type=str, default=20,
                        help='threshold magnitude for np.int16 dtype')
    args, _ = parser.parse_known_args()

    #test_threshold(args)
    split_wavs(args)

    parser = argparse.ArgumentParser(description='Audio Classification Training')
    parser.add_argument('--model_fn', type=str, default='models/lstm.h5',
                        help='model file to make predictions')
    parser.add_argument('--pred_fn', type=str, default='y_pred',
                        help='fn to write predictions in logs dir')
    parser.add_argument('--src_dir', type=str, default='Random Song/Testing',
                        help='directory containing wavfiles to predict')
    parser.add_argument('--dt', type=float, default=1.0,
                        help='time in seconds to sample audio')
    parser.add_argument('--sr', type=int, default=16000,
                        help='sample rate of clean audio')
    parser.add_argument('--threshold', type=str, default=20,
                        help='threshold magnitude for np.int16 dtype')
    args, _ = parser.parse_known_args()

    make_prediction(args)

