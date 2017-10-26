from time import sleep
import numpy as np
from PIL import Image

from catchertest import CatcherforTest

from nec_agent import NECAgent
#import tensorflow as tf

# def image_preprocessor(state, size=(40, 40)):
#     #state = state[32:195, :, :]
#     #state = misc.imresize(state, size)
#     # grayscaling and normalizing state
#     state = np.dot(state[..., :3], np.array([0.299, 0.587, 0.114], dtype=np.float32)) / 255.0
#     return state

nec_agent_parameters_dict = {
    "log_save_directory": "C:/Work/temp/nec_agent",
    "dnd_max_memory": 50000,
    "input_shape": (40, 40, 2),
    "fully_conn_neurons": 64,
    "neighbor_number": 25,
    "num_outputs": (16, 16),
    "n_step_horizon": 10,
    "kernel_size": ((3, 3), (3, 3)),
    "stride": ((2, 2), (2, 2)),
    "batch_size": 32,
    "epsilon_decay_bounds": (3000, 20000),
    "backprop_learning_rate": 1e-4,
    "tabular_learning_rate": 1e-3

}

agent = NECAgent([0, 2, 3], **nec_agent_parameters_dict)
#tf.summary.FileWriter("C:/RL/nec_saves", graph=agent.session.graph)

max_ep_num = 500000

env = CatcherforTest()

games_reward_list = []
reward = None

for i in range(max_ep_num):
    done = False

    observation = env.reset()
    processed_obs = observation  # nincs benne az image preproc

    if i % 100 == 0:
        print("Elkezdődött játék sorszáma: ", i+1)
        print("Összes lépés: ", (i+1)*18)

    while not done:
        action = agent.get_action(processed_obs)

        observation, reward, done = env.step(action)
        agent.save_action_and_reward(action, reward)

        processed_obs = observation

    agent.update()
    agent.reset_episode_related_containers()

    games_reward_list.append(reward)
    if i % 100 == 0:
        print("Ucso 100 game reward összege: ", sum(games_reward_list))
        games_reward_list = []
