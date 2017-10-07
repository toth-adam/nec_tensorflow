from collections import deque
import numpy as np


class ReplayMemory:
    def __init__(self, size, stack_size):
        self.stack_size = stack_size
        self.rep_mem = deque(maxlen=int(size))
        self.episode_end = deque(maxlen=int(size))

    def append(self, item, ep_end):
        self.rep_mem.append(item)
        self.episode_end.append(ep_end)

    def get_batch(self, batch_size):
        rand_samp_num = []
        while len(rand_samp_num) < batch_size:
            r_ind = np.random.randint(len(self.rep_mem))
            if r_ind not in rand_samp_num and not self.episode_end[r_ind]:
                rand_samp_num.append(r_ind)

        rand_samp_list = []
        for rand_index in rand_samp_num:
            stacked_frames = []
            seen_false = False
            false_state = None
            for i in range(self.stack_size):
                if self.episode_end[rand_index - i] is True:
                    seen_false = True
                    false_state = self.rep_mem[rand_index - i][0]
                if not seen_false:
                    stacked_frames.append(self.rep_mem[rand_index - i][0])
                else:
                    stacked_frames.append(false_state)

            numpy_appended_frames = np.asarray(stacked_frames)
            numpy_stacked_frames = np.stack(numpy_appended_frames, axis=2)
            rand_samp_list.append([numpy_stacked_frames, self.rep_mem[rand_index][1], self.rep_mem[rand_index][2]])

        trans_rand_sample = list(map(list, zip(*rand_samp_list)))
        return trans_rand_sample[0], trans_rand_sample[1], trans_rand_sample[2]  # state, action, q_n
