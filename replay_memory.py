from lru import LRU
import numpy as np

class replay_memory:

    def __init__(self, size=5e5):
        self.rep_mem = LRU(int(size))
        self.size = size

    def append(self, item):
        if len(self.rep_mem) < self.size:
            num = len(self.rep_mem) #increment the key's number
        else:
            # get the last element's key and in the next step overwrite its value
            num = self.rep_mem.peek_last_item()[0]
        self.rep_mem[num] = item

    def get(self, batch_size):
        sample_list = []
        # kirandomol batch size darab keyt
        rand_samp_num = np.random.choice(range(len(self.rep_mem)), size=batch_size, replace=False)
        # kiveszi a kirandomolt keyekhez tartozó értékeket az LRU-ból
        for i in rand_samp_num:
            sample_list.append(self.rep_mem[i])
        return sample_list