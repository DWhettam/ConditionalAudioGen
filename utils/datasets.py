from __future__ import print_function
import librosa
import librosa.filters
import dcase_util
import librosa
import librosa.display
import pandas as pd
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import os

def make_log_spectrogram(audio_array):
    n_fft = 2048
    hop_length = 512
    win_length = n_fft

    stft = librosa.stft(audio_array, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window='hann',
                        center=True,
                        pad_mode='reflect')
    stft = np.log10(stft + 1)
    return stft


class AudioDataset(Dataset):
    def __init__(self, root_path, csv_path):
        if not os.path.exists(root_path) or not os.path.exists(csv_path):
            raise Exception('path does not exist')
        self.root_path = root_path
        self.csv_path = csv_path
        self.data = pd.read_csv(self.csv_path, sep='\t')
        self.data_len = len(self.data)
        self.process_labels()

    def __getitem__(self, index):
        file_wav = self.root_path + self.data.iloc[index]['filename']
        label = self.data.iloc[index]['scene_label']
        y, sr = librosa.load(file_wav)
        spectrogram = make_log_spectrogram(y)

        spectrogram = np.expand_dims(spectrogram, 0)
        spectrogram = spectrogram.astype(np.float32)
        spectrogram = torch.from_numpy(spectrogram)
        label = np.asarray(label)
        label = torch.from_numpy(label).long()

        return spectrogram, label

    def __len__(self):
        return self.data_len

    def process_labels(self):
        self.data.scene_label = pd.Categorical(self.data.scene_label)
        self.data['scene_label'] = self.data.scene_label.cat.codes

def get_data_loader():
    # dataset = dcase_util.datasets.TAUUrbanAcousticScenes_2019_DevelopmentSet(data_path=path)
    # dataset.initialize()
    path = os.getcwd() + '/../scratch/BlindCamera/TAU-urban-acoustic-scenes-2019-development/'
    train_csv = path + 'evaluation_setup/fold1_train.csv'

    TRAIN = AudioDataset(path, train_csv)
    TRAIN_LOADER = DataLoader(dataset=TRAIN, batch_size=32, shuffle=True)

    return TRAIN_LOADER

