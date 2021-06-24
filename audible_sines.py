import pytorch_lightning as pl
import torch
import numpy as np
import librosa
import random
import math

class SinePoolBase(torch.utils.data.IterableDataset):
    @staticmethod
    def db_to_amp(db):
        return 10**(db/20)

    def __init__(self, sr=44100, sample_len=2, fade_len=0.2):
        super().__init__()
        self.sr = sr
        self.sample_len = int(sr * sample_len)
        fade_samples = int(sr * fade_len)
        self.window = np.concatenate(
            (np.linspace(0,1,fade_samples), 
            np.ones(self.sample_len - 2*fade_samples), 
            np.linspace(1,0,fade_samples)))

        assert len(self.window) == self.sample_len

class SinePoolDeterministic(SinePoolBase):
    def __init__(self, freqs: list[float]=None, volumes: list[float]=None, **kwargs) -> None:
        super().__init__(**kwargs)
        if freqs is None:
            min = math.log(20)
            max = math.log(20000)
            steps = 20
            self.freqs = [math.exp(i * (max - min) / steps + min) for i in range(steps)]
        else:
            self.freqs = freqs

        self.volumes = volumes if volumes is not None else [0, -3, -9, -20, -30]

    def __iter__(self):
        for freq in self.freqs:
            for vol in self.volumes:
                tone = librosa.tone(freq, sr=self.sr, length=self.sample_len) * self.db_to_amp(vol)
                yield tone * self.window

class SinePool(SinePoolBase):
    def __init__(self, epoch_size=100, min_freq=20, max_freq=20000, min_volume=-14, max_volume=0, 
            mixup_alpha=0.05, **kwargs) -> None:
        super().__init__(**kwargs)
        self.epoch_size = epoch_size
        self.min_freq = math.log(min_freq)
        self.max_freq = math.log(max_freq)
        self.min_volume = min_volume
        self.max_volume = max_volume
        self.mixup_alpha = mixup_alpha

    def mixup(self, x1, x2):
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        return x1 * lam + x2 * (1 - lam)

    def __iter__(self):
        for i in range(self.epoch_size):
            yield self.mixup(self.random_sine(), self.random_sine())

    @staticmethod
    def scale(x, low, high):
        return low + x * (high - low)

    def random_sine(self):
        freq = math.exp(self.scale(random.weibullvariate(0.25, 1.48), self.min_freq, self.max_freq))
        amp = self.db_to_amp(random.uniform(self.min_volume, self.max_volume))
        tone = librosa.tone(freq, sr=self.sr, length=self.sample_len) * amp
        return tone * self.window

class AudibleSines(pl.LightningDataModule):
    pass