import numpy as np
from scipy import misc
import time
from collections import deque
import threading
import tkinter as Tk
import os

import gym

from nec_agent import NECAgent, setup_logging


class GameScene():
    def __init__(self, gui_message_queue, gst_message_queue):
        self.gui_message_queue = gui_message_queue
        self.gst_message_queue = gst_message_queue

        self.tensorboard_log = "/media/david/bigdaddy/RL/pacman_2017_12_18_DND250/run_2"

        nec_agent_parameters_dict = {
            "dnd_max_memory": 250000,
            "log_save_directory": self.tensorboard_log,
            "input_shape": (42, 42, 4),
            "kernel_size": ((3, 3), (3, 3), (3, 3), (3, 3)),
            "num_outputs": (32, 32, 32, 32),
            "stride": ((2, 2), (2, 2), (2, 2), (2, 2)),
            "batch_size": 32,
            "fully_conn_neurons": 128,
            "tabular_learning_rate": 0.5
        }

        self.agent = NECAgent([1, 2, 3, 4], **nec_agent_parameters_dict)

        self.agent.full_load(self.tensorboard_log, 1123723)

        self.max_ep_num = 50000

        self.env = gym.make('MsPacman-v4')
        # env.env.frameskip = 4
        self.games_reward_list = []
        self.games_step_num_list = []
        self.game_step_number = 0
        self.last_save_time = time.time()

        self.shouldRender = False


    def image_preprocessor(self, state, size=(42, 42)):
        state = state[:173, :, :]
        state = misc.imresize(state, size)
        # greyscaling and normalizing state
        state = np.dot(state[..., :3], np.array([0.299, 0.587, 0.114], dtype=np.float32)) / 255.0
        return state


    # setup_logging()


    def start(self):
        # setup_logging(
        #     file_handler_filename="/media/david/bigdaddy/RL/pacman_2017_12_18_DND250/run_1/log_1.txt",
        #     is_file_handler=True)
        for i in range(self.max_ep_num):

            env = self.env

            _ = env.reset()

            env.env.frameskip = 265
            observation, reward, done, info = env.step(0)
            env.env.frameskip = (2, 5)

            processed_obs = self.image_preprocessor(observation)
            sum_reward = 0

            while not done:

                # Check if there is message from the GUI
                if self.gst_message_queue:
                    msg = self.gst_message_queue.pop()
                    if msg == 2:
                        # os.system(str("tensorboard --logdir " + self.tensorboard_log))
                        print("azthitted, majdlegközelebb")
                    elif not self.shouldRender:
                        self.shouldRender = True
                    else:
                        self.shouldRender = False
                        env.render(close=True)

                if self.shouldRender:
                    env.render()

                action = self.agent.get_action(processed_obs)
                observation, reward, done, info = env.step(action)
                self.agent.save_action_and_reward(action, reward)

                processed_obs = self.image_preprocessor(observation)

                self.game_step_number += 1
                sum_reward += reward

                if done:
                    self.agent.update()

                    self.games_reward_list.append(sum_reward)
                    self.games_step_num_list.append(self.game_step_number)
                    self.game_step_number = 0

            # For logging purposes
            if i % 10 == 0:
                print()
                print("Number of total games: ", i + 1)
                print("Total step numbers: ", self.agent.global_step)
                print()
                print("Mean step number: ", np.mean(self.games_step_num_list))
                print()
                print("Games' rewards list:")
                print(self.games_reward_list)
                #    print("Previously seen states number:", agent.seen_states_number)
                # print("Current Alpha is: {}".format(agent.curr_alpha()))
                print("-----------------------------------------------------------------------------")

                self.games_reward_list = []
                self.games_step_num_list = []

            if time.time() - self.last_save_time > 9000:
                self.agent.full_save(self.tensorboard_log)
                self.last_save_time = time.time()
                print("/////////////////////////////////////// SAVE ////////////////////////////////////////")


class GameSceneThread(threading.Thread):
    def __init__(self, gui_message_queue, gst_message_queue):
        self.queue1 = gui_message_queue
        self.queue2 = gst_message_queue
        threading.Thread.__init__(self, daemon=True)

    def run(self):
        gs = GameScene(self.queue1, self.queue2)
        gs.start()
        return


class GuiPacman:
    def __init__(self, master):
        self.master = master
        master.title("Pacman")
        self.button = Tk.Button(self.master, command=self.click_start)
        self.button.configure(text="Start", background="Grey", padx=50)
        self.button.pack(side="top")

        self.button3 = Tk.Button(self.master, command=self.start_tensorboard)
        self.button3.configure(text="Tensorboard", background="Grey", padx=50)
        self.button3.pack(side="bottom")

        self.gui_message_queue = deque()
        self.gst_message_queue = deque()

        self.master.after(100, self.process_queue)
        self.gst = None


    def click_start(self):
        if not self.gst:
            self.gst = GameSceneThread(self.gui_message_queue, self.gst_message_queue)
            self.button2 = Tk.Button(self.master, command=self.click_render)
            self.button2.configure(text="Render", background="Green", padx=50)
            self.button2.pack(side="right")
            self.gst.start()

    def click_render(self):
        self.gst_message_queue.append(1)

    def start_tensorboard(self):
        self.gst_message_queue.append(2)

    def process_queue(self):
        if self.gui_message_queue:
            print("processing message")
            msg = self.gui_message_queue.pop()
            print(str(msg))
            self.master.after(300, self.process_queue)
        else:
            # print("No message, but the thread is alive")
            self.master.after(300, self.process_queue)


root = Tk.Tk()
ui = GuiPacman(root)
root.mainloop()
