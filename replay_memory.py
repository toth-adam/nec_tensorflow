from collections import deque
import numpy as np


class ReplayMemory:
    def __init__(self, size=5e5):
        self.rep_mem = deque(maxlen=size)
        self.size = size

    def append(self, item):
        self.rep_mem.append(item)

    def get_batch(self, batch_size):
        sample_list = []
        # kirandomol batch size darab sorszámot
        rand_samp_num = np.random.choice(range(self.size), size=batch_size, replace=False)
        # kiveszi a kirandomolt sorszámhoz tartozó értékeket az LRU-ból
        for i in rand_samp_num:
            sample_list.append(self.rep_mem[i])
        return np.asarray(sample_list)
