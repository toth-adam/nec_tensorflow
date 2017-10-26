import numpy as np


class CatcherforTest:
    def __init__(self):

        self.ball_init_col = np.random.randint(0, 37)
        self.ball_init_row = 0
        self.pad_init_col = np.random.randint(0, 30)
        self.pad_row = 38
        self.blank_map = np.zeros((40, 40))
        self.ball = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]])
        self.pad = []
        for i in range(2):
            for j in range(10):
                self.pad.append([i, j])
        self.done = False

    def add_ball_pad(self):
        pass

    def reset(self):
        self.ball_init_col = np.random.randint(0, 37)
        self.ball_init_row = 0
        self.pad_init_col = np.random.randint(10, 20)
        self.blank_map = np.zeros((40, 40))

        for i in self.ball:
            self.blank_map[i[0] + self.ball_init_row][i[1] + self.ball_init_col] = 1
        for i in self.pad:
            self.blank_map[i[0] + self.pad_row][i[1] + self.pad_init_col] = 1

        self.done = False

        return np.array(self.blank_map, copy=True, dtype=np.float32)

    def step(self, action):
        reward = 0
        self.ball_init_row += 2
        if action == 2:
            # reward = -0.05
            if self.pad_init_col < 30:
                self.pad_init_col += 1
        elif action == 0:
            # reward = -0.05
            if self.pad_init_col > 0:
                self.pad_init_col -= 1

        if self.ball_init_row >= 36:
            self.done = True
            if self.ball_init_col >= self.pad_init_col and self.ball_init_col <= self.pad_init_col + 7:
                reward = 1
            else:
                reward = -1

        if self.done is False:
            self.blank_map = np.zeros((40, 40))

            for i in self.ball:
                self.blank_map[i[0] + self.ball_init_row][i[1] + self.ball_init_col] = 1
            for i in self.pad:
                self.blank_map[i[0] + self.pad_row][i[1] + self.pad_init_col] = 1

        return np.array(self.blank_map, copy=True, dtype=np.float32), reward, self.done
