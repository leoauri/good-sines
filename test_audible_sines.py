from audible_sines import SinePool, SinePoolDeterministic, AudibleSines
import math
import numpy as np
from torch import Tensor

sine_pool = SinePool()

def test_epoch_size():
    assert len(list(iter(sine_pool))) == sine_pool.epoch_size

def test_contains_something():
    sine = sine_pool.random_sine()
    assert sine.max() != sine.min()

def amp_to_db(amp):
    return 20 * math.log(amp, 10)

def test_volumes():
    for vol in [-30, -20, -12, -6, -3, 0]:
        sine_pool_volume = SinePool(min_volume=vol, max_volume=vol)
        sine = sine_pool_volume.random_sine()
        assert amp_to_db(sine.max()) - vol < 1e-8

def test_volume_stochastic():
    sine_pool_volume = SinePool(min_volume=-6, max_volume=-3)
    count = 0
    for sine in sine_pool_volume:
        assert amp_to_db(sine.max()) < -3 + 1e-8
        assert amp_to_db(sine.max()) > -6 - 1e-8
        count += 1
    assert count == sine_pool_volume.epoch_size

def test_deterministic_sine_pool():
    dsp = SinePoolDeterministic()
    assert len(list(dsp)) == len(dsp.freqs) * len(dsp.volumes)
    for s in dsp:
        assert s.min() != s.max()

def test_window():
    pools = [SinePoolDeterministic(volumes=[0]), SinePool(min_volume=0, max_volume=0, epoch_size=10)]
    for pool in pools:
        for s in pool:
            assert np.all(np.less_equal(abs(s), pool.window))

def test_datamodule():
    dm = AudibleSines()
    dm.prepare_data()
    dm.setup(stage='fit')

    for batch in dm.train_dataloader():
        assert batch.shape == (4, dm.train_set.sample_len)
        assert isinstance(batch, Tensor)
        assert isinstance(batch[0], Tensor)