import numpy as np


class CatcherforTest:
    def __init__(self, map_size=40, ball_size=3, pad_size=10):

        self.map_size = map_size
        self.ball_size = ball_size
        self.pad_size = pad_size
        self.blank_map = None
        self.b_coord = None
        self.p_coord = None
        self.done = False
        self.ball_color = np.array([200, 20, 20], dtype=np.int32)
        self.pad_color = np.array([50, 200, 50], dtype=np.int32)

        self.p_1 = np.asarray(np.stack((np.zeros(self.pad_size), np.arange(self.pad_size)), axis=1), dtype=np.int32)
        self.p_2 = np.asarray(np.stack((np.ones(self.pad_size), np.arange(self.pad_size)), axis=1), dtype=np.int32)
        self.pad = np.vstack((self.p_1, self.p_2))

        self.ball = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]])

    def reset(self):
        self.b_coord = [0, np.random.randint(1, self.map_size - 4)]
        self.p_coord = [38, np.random.randint(1, self.map_size - self.pad_size)]
        self.blank_map = np.asarray([[100, 150, 200]] * (self.map_size ** 2), dtype=np.int32).reshape((40, 40, 3))
        self.done = False
        actual_ball_coords = self.ball + self.b_coord
        actual_pad_coords = self.pad + self.p_coord
        for a_b_c in actual_ball_coords:
            self.blank_map[a_b_c[0], a_b_c[1]] = self.ball_color

        for a_p_c in actual_pad_coords:
            self.blank_map[a_p_c[0], a_p_c[1]] = self.pad_color

        return np.array(self.blank_map, copy=True)

    def step(self, action):
        reward = 0
        self.b_coord = self.b_coord + np.array([2, 0], dtype=np.int32)
        if action == 2:
            #reward = -0.05
            if self.p_coord[1] < 30:
                self.p_coord = self.p_coord + np.array([0, 1], dtype=np.int32)
        elif action == 0:
            #reward = -0.05
            if self.p_coord[1] > 0:
                self.p_coord = self.p_coord + np.array([0, -1], dtype=np.int32)

        if self.b_coord[0] >= 36:
            self.done = True
            if self.b_coord[1] >= self.p_coord[1] and self.b_coord[1] <= self.p_coord[1] + 10:
                reward = 1
            else:
                reward = -1

        if self.done == False:
            actual_ball_coords = self.ball + self.b_coord
            actual_pad_coords = self.pad + self.p_coord
            self.blank_map = np.asarray([[100, 150, 200]] * (self.map_size ** 2), dtype=np.int32).reshape((40, 40, 3))

            for a_b_c in actual_ball_coords:
                self.blank_map[a_b_c[0], a_b_c[1]] = self.ball_color

            for a_p_c in actual_pad_coords:
                self.blank_map[a_p_c[0], a_p_c[1]] = self.pad_color

        return np.array(self.blank_map, copy=True), reward, self.done