from .base import BaseChannel


class BernoulliChannel(BaseChannel):
    def __init__(self, p=(0.5, )):
        super(BernoulliChannel, self).__init__()
        self.num_of_errors = len(p)
        self.p = p

    def noise_block(self, x):
        pass
