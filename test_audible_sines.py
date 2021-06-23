from audible_sines import SinePool
import math

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
