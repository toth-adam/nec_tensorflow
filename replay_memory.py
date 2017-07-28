from lru import LRU
import numpy as np

class replay_memory:

    def __init__(self, size=5e5):
        self.rep_mem = LRU(int(size)) # size should be "int", kiírta hogy int kell neki ha csak simán bebasztam a size-t (dunno miért)

    def append(self, item):
        self.rep_mem[len(self.rep_mem)] = item # simple item appending

    def get(self, batch_size):
        sample_list = []
        # kirandomol batch size darab sorszámot
        rand_samp_num = np.random.choice(range(len(self.rep_mem)), size=batch_size, replace=False)
        # kiveszi a kirandomolt sorszámhoz tartozó értékeket az LRU-ból
        for i in rand_samp_num:
            sample_list.append(self.rep_mem[i])
        return sample_list