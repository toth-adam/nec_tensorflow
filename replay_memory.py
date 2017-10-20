from collections import deque
import numpy as np


class ReplayMemory:
    def __init__(self, size=1e5):
        self.rep_mem = deque(maxlen=int(size))

    def append(self, item):
        self.rep_mem.append(item)

    def get_batch(self, batch_size):
        rand_samp_num = np.random.choice(len(self.rep_mem), size=batch_size, replace=False)
        rand_samp_list = [self.rep_mem[i] for i in rand_samp_num]
        trans_rand_sample = list(map(list, zip(*rand_samp_list)))
        return trans_rand_sample[0], trans_rand_sample[1], trans_rand_sample[2]  # state, action, q_n

    def save(self, path, glob_step_num):
        np.save(path + '/rep_mem_' + str(glob_step_num) + '.npy', self.rep_mem)
        # np.save(path + '/episode_end_' + str(glob_step_num) + '.npy', self.episode_end)

    def load(self, path, glob_step_num):
        r_m = np.load(path + '/rep_mem_' + str(glob_step_num) + '.npy')
        # e_e = np.load(path + '/episode_end_' + str(glob_step_num) + '.npy')
        for r_m_i in r_m:
            self.append(r_m_i)
