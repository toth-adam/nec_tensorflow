import numpy as np
from scipy import misc
import time

import gym

from nec_agent import NECAgent, setup_logging


def image_preprocessor(state, size=(42, 42)):
    state = state[32:195, :, :]
    state = misc.imresize(state, size)
    # greyscaling and normalizing state
    state = np.dot(state[..., :3], np.array([0.299, 0.587, 0.114], dtype=np.float32)) / 255.0
    return state


setup_logging()

nec_agent_parameters_dict = {
    # "log_save_directory": "C:/RL/nec_saves",
    "dnd_max_memory": 150000,
    "input_shape": (42, 42, 3),
    "kernel_size": ((3, 3), (3, 3), (3, 3)),
    "num_outputs": (16, 16, 16),
    "neighbor_number": 50,
    "epsilon_decay_bounds": (5000, 25000),
    "tab_update_for_neighbours_dist": 0.0022,
    "stride": ((2, 2), (2, 2), (2, 2))
}

agent = NECAgent([0, 2, 3], **nec_agent_parameters_dict)

#agent.full_load("D:/RL/nec_saves", 1170862)

max_ep_num = 5

env = gym.make('Pong-v4')

games_reward_list = []
games_step_num_list = []
game_step_number = 0
last_save_time = time.time()

for i in range(max_ep_num):
    start_time = time.time()

    done = False
    mini_game_done = False

    observation = env.reset()
    processed_obs = image_preprocessor(observation)

    while not done:
        action = agent.get_action(processed_obs)

        observation, reward, done, info = env.step(action)
        agent.save_action_and_reward(action, reward)

        # mini_game_done változó beállítása, mert a gym env 21 pong játékot vesz egy játéknak
        mini_game_done = True if abs(reward) == 1 else False

        processed_obs = image_preprocessor(observation)

        game_step_number += 1

        if mini_game_done:
            agent.update()

            games_reward_list.append(reward)
            games_step_num_list.append(game_step_number)
            game_step_number = 0

    # For logging purposes
    unique, counts = np.unique(games_reward_list, return_counts=True)

    print()
    print("Number of total games: ", i+1)
    print("Total step numbers: ", agent.global_step)
    print()
    print("Number of games: ", len(games_reward_list))
    print("Number of won (1) and lost (-1) games: ", dict(zip(unique, counts)))
    print("Mean step number: ", np.mean(games_step_num_list))
    print()
    print("Games' rewards list:")
    print(games_reward_list)
    print()
    print("Games' steps number:")
    print(games_step_num_list)
    print("Previously seen states number:", agent.seen_states_number)
    print("-----------------------------------------------------------------------------")


    games_reward_list = []
    games_step_num_list = []

    if time.time() - last_save_time > 10800:
        agent.full_save("D:/RL/nec_saves")
        last_save_time = time.time()
        print("/////////////////////////////////////// SAVE ////////////////////////////////////////")

