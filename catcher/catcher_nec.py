# from time import sleep
import numpy as np
# from PIL import Image
from scipy import misc
from catcher import CatcherforTest
# import matplotlib.pyplot as plt

from nec_agent import NECAgent, setup_logging
# import tensorflow as tf

def image_preprocessor(state, size=(20, 20)):
    # state = state[32:195, :, :]
    state = misc.imresize(state, size)
    # greyscaling and normalizing state
    state = np.dot(state[..., :3], np.array([0.299, 0.587, 0.114], dtype=np.float32)) / 255.0
    return state

nec_agent_parameters_dict = {
    # "log_save_directory": "C:/RL/NEC",
    "dnd_max_memory": 15000,
    "input_shape": (20, 20, 2),
    "kernel_size": ((3, 3), (3, 3)),
    "num_outputs": (32, 32),
    "stride": ((2, 2), (2, 2)),
    "batch_size": 32,
    "fully_conn_neurons": 32,
    "tabular_learning_rate": 0.5,
    "neighbor_number": 15,
    "epsilon_decay_bounds": (2000, 15000)

}
setup_logging()
agent = NECAgent([0, 2, 3], **nec_agent_parameters_dict)
# agent.full_load("C:/RL/NEC/Full_save", 1800)

max_ep_num = 500000

env = CatcherforTest()

games_reward_list = []
reward = None

for i in range(max_ep_num):
    done = False

    observation = env.reset()
    processed_obs = image_preprocessor(observation)  # nincs benne az image preproc

    # plt.imshow(np.asarray(observation, dtype=np.float32))
    # plt.show()
    # plt.imshow(processed_obs, cmap="gray")
    # plt.show()
    if i % 50 == 0 and i != 0:
        print("Elkezdődött játék sorszáma: ", i+1)
        print("Összes lépés: ", (i+1)*18)

    # if i % 100 == 0 and i != 0:
    #     agent.full_save("C:/RL/NEC/Full_save")
    #     print("mentett")
    #     break

    while not done:
        action = agent.get_action(processed_obs)

        observation, reward, done = env.step(action)
        agent.save_action_and_reward(action, reward)

        processed_obs = image_preprocessor(observation)
        # plt.imshow(processed_obs, cmap="gray")
        # plt.show()

    agent.update()
    agent.reset_episode_related_containers()

    games_reward_list.append(reward)
    if i % 50 == 0:
        print("Ucso 50 game reward összege: ", sum(games_reward_list))
        games_reward_list = []
