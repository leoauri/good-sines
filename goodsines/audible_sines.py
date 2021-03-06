from typing import Optional, List
import pytorch_lightning as pl
import torch
import numpy as np
import librosa
import random
import math
from torch.utils.data import DataLoader
from functools import partial


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
    def __init__(self, freqs: List[float]=None, volumes: List[float]=None, **kwargs) -> None:
        super().__init__(**kwargs)
        if freqs is None:
            minf = math.log(20)
            maxf = math.log(20000)
            steps = 20
            self.freqs = [math.exp(i * (maxf - minf) / steps + minf) for i in range(steps)]
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
    def __init__(self, batch_size=4, train_set=None, valid_set=None, test_set=None):
        super().__init__()
        self.batch_size = batch_size
        self.train_set = SinePool if train_set is None else train_set
        self.valid_set = SinePoolDeterministic if valid_set is None else valid_set
        self.test_set = partial(SinePoolDeterministic, freqs=[60, 120, 600, 2000], 
            volumes=[0, -12, -14]) if test_set is None else test_set

    def setup(self, stage: Optional[str] = None):
        self.train_set = self.train_set()
        self.valid_set = self.valid_set()
        self.test_set = self.test_set()

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.valid_set, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size)


class SineTo8vb(SinePoolDeterministic):
    def __init__(self, phases=None, **kwargs):
        super().__init__(**kwargs)
        self.phases = ['random']*50 if phases is None else phases

    def dataset_pair(self, vol='random', phi='random'):
        phi_calculated = random.uniform(0, math.pi*2) if phi == 'random' else phi
        vol_calculated = (random.weibullvariate(0.25, 1.48) * -40 if vol == 'random' 
                else vol)
        x = librosa.tone(400, sr=self.sr, length=self.sample_len, 
            phi=phi_calculated) * self.db_to_amp(vol_calculated)
        y = librosa.tone(200, sr=self.sr, length=self.sample_len, 
            phi=random.uniform(0, math.pi*2)) * self.db_to_amp(vol_calculated)
        return torch.Tensor(x * self.window), torch.Tensor(y * self.window)

    def __iter__(self):
        for vol in self.volumes:
            for phi in self.phases:
                yield self.dataset_pair(vol, phi)

SineTo8vbValid = partial(SineTo8vb, phases=np.linspace(0, math.pi*2,
        endpoint=False))

SineTo8vbDataModule = partial(AudibleSines, train_set=SineTo8vb,
        valid_set=SineTo8vbValid)
