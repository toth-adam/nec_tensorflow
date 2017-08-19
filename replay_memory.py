from collections import deque
import numpy as np


class ReplayMemory:
    def __init__(self, size=5e5):
        self.rep_mem = deque(maxlen=size)

    def append(self, item):
        self.rep_mem.append(item)

    def get_batch(self, batch_size):
        rand_samp = np.random.choice(self.rep_mem, size=batch_size, replace=False)
        return np.asarray(rand_samp)
