import pytorch_lightning as pl
import torch
import numpy as np
import librosa
import random

class SinePool(torch.utils.data.IterableDataset):
    def __init__(self, epoch_size=100, sr=44100, sample_len=2, fade_len=0.2, min_freq=20, 
            max_freq=20000):
        super(SinePool).__init__()
        self.epoch_size = epoch_size
        self.sr = sr
        self.sample_len = int(sr * sample_len)
        fade_samples = int(sr * fade_len)
        self.window = np.concatenate(
            (np.linspace(0,1,fade_samples), 
            np.ones(self.sample_len - 2*fade_samples), 
            np.linspace(1,0,fade_samples)))
        self.min_freq = min_freq
        self.max_freq = max_freq

        assert len(self.window) == self.sample_len

    def __iter__(self):
        for i in range(self.epoch_size):
            yield self.random_sine()

    def random_sine(self):
        freq = random.uniform(self.min_freq, self.max_freq)
        tone = librosa.tone(freq, sr=self.sr, length=self.sample_len)
        return tone * self.window

class AudibleSines(pl.LightningDataModule):
    pass