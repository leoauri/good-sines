from audible_sines import SinePool

sine_pool = SinePool()

def test_epoch_size():
    assert len(list(iter(sine_pool))) == sine_pool.epoch_size
