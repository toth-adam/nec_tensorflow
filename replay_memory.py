from collections import deque
import numpy as np
import os


class ReplayMemory:
    def __init__(self, size, stack_size):
        self.stack_size = stack_size
        self.size = int(size)
        self.maxlen = int(size + stack_size - 1)
        self.rep_mem = deque(maxlen=self.maxlen)
        self.episode_end = deque(maxlen=self.maxlen)
        self.choice_range = np.arange(stack_size - 1, size + stack_size - 1)

    def append(self, item, ep_end):
        self.rep_mem.append(item)
        self.episode_end.append(ep_end)

    def get_batch(self, batch_size):
        if len(self.rep_mem) == self.maxlen:
            rand_samp_num = np.random.choice(self.choice_range, batch_size, replace=False).astype(np.int32)
        else:
            rand_samp_num = np.random.choice(len(self.rep_mem), batch_size, replace=False).astype(np.int32)

        batch_states = []
        batch_actions = []
        batch_q_ns = []
        for rand_index in rand_samp_num:
            stacked_frames = []
            for i in range(self.stack_size):
                if (self.episode_end[rand_index - i] is True and i > 0) or rand_index - i < 0:
                    last_false_state = [self.rep_mem[rand_index - i + 1][0]] * (self.stack_size - i)
                    stacked_frames = stacked_frames + last_false_state
                    break
                else:
                    stacked_frames.append(self.rep_mem[rand_index - i][0])

            numpy_appended_frames = np.asarray(stacked_frames)
            # reversed the order of frame order
            numpy_stacked_frames = np.stack(numpy_appended_frames[::-1, ...], axis=2)

            batch_states.append(numpy_stacked_frames)
            batch_actions.append(self.rep_mem[rand_index][1])
            batch_q_ns.append(self.rep_mem[rand_index][2])

        return (np.array(batch_states, dtype=np.float32),  np.array(batch_actions, dtype=np.int32),
                np.array(batch_q_ns, dtype=np.float32))

    def save(self, path, glob_step_num):
        try:
            os.mkdir(path + '/replay_memory_' + str(glob_step_num))
        except FileExistsError:
            pass
        np.save(path + '/replay_memory_' + str(glob_step_num) + '/memory.npy', self.rep_mem)
        np.save(path + '/replay_memory_' + str(glob_step_num) + '/episode_end.npy', self.episode_end)

    def load(self, path, glob_step_num):
        r_m = np.load(path + '/replay_memory_' + str(glob_step_num) + '/memory.npy')
        e_e = np.load(path + '/replay_memory_' + str(glob_step_num) + '/episode_end.npy')

        for r_m_i, e_e_i in zip(r_m, e_e):
            self.append(r_m_i, e_e_i)
