from collections import deque
import numpy as np


class ReplayMemory:
    def __init__(self, size=1e5):
        self.rep_mem = deque(maxlen=int(size))

    def append(self, item):
        self.rep_mem.append(item)

    def get_batch(self, batch_size):
        rand_samp_list = []
        rand_samp_num = np.random.choice(len(self.rep_mem), size=batch_size, replace=False)
        for i in rand_samp_num:
            rand_samp_list.append(self.rep_mem[i])
        trans_rand_sample = np.asarray(rand_samp_list).T
        return trans_rand_sample[0], trans_rand_sample[1], trans_rand_sample[2] #  state, action, q_n
