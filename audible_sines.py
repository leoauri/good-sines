import pytorch_lightning as pl
import torch
import numpy as np
import librosa
import random

class SinePool(torch.utils.data.IterableDataset):
    def __init__(self, epoch_size=100, sr=44100, sample_len=2, fade_len=0.2, min_freq=20, 
            max_freq=20000, min_volume=-40, max_volume=0):
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
        self.min_volume = min_volume
        self.max_volume = max_volume

        assert len(self.window) == self.sample_len

    def __iter__(self):
        for i in range(self.epoch_size):
            yield self.random_sine()

    @staticmethod
    def db_to_amp(db):
        return 10**(db/20)

    def random_sine(self):
        freq = random.uniform(self.min_freq, self.max_freq)
        amp = self.db_to_amp(random.uniform(self.min_volume, self.max_volume))
        tone = librosa.tone(freq, sr=self.sr, length=self.sample_len) * amp
        return tone * self.window

class AudibleSines(pl.LightningDataModule):
    pass