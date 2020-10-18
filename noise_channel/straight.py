from .base import BaseChannel


class StraightChannel(BaseChannel):
    def noise_block(self, x):
        assert x.shape[0] == self.code_size
        return x
