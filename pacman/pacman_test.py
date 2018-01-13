import numpy as np
from scipy import misc
import time

import gym

from nec_agent import NECAgent, setup_logging


def image_preprocessor(state, size=(42, 42)):
    state = state[:173, :, :]
    state = misc.imresize(state, size)
    # greyscaling and normalizing state
    state = np.dot(state[..., :3], np.array([0.299, 0.587, 0.114], dtype=np.float32)) / 255.0
    return state


# setup_logging(file_handler_filename="/media/david/df8ae8cc-4282-4f1f-abce-0b066034555d/david/RL/pacman_2017_12_07/run_4/log.txt")
# setup_logging()

# save_dir = "/media/david/df8ae8cc-4282-4f1f-abce-0b066034555d/david/RL/pacman_2017_12_07/run_4"

nec_agent_parameters_dict = {
    "cpu_only": True,
    "dnd_max_memory": 250000,
    "input_shape": (42, 42, 4),
    "kernel_size": ((3, 3), (3, 3), (3, 3), (3, 3)),
    "num_outputs": (32, 32, 32, 32),
    "stride": ((2, 2), (2, 2), (2, 2), (2, 2)),
    "batch_size": 32,
    "fully_conn_neurons": 128,
    "tabular_learning_rate": 0.5
}

agent = NECAgent([1, 2, 3, 4], **nec_agent_parameters_dict)

agent.full_load("/media/david/bigdaddy/RL/pacman_2017_12_22_DND250_tab/run_4", 1207300)

max_ep_num = 50000

env = gym.make('MsPacman-v4')
# env.env.frameskip = 4
games_reward_list = []
games_step_num_list = []
game_step_number = 0
last_save_time = time.time()

for i in range(max_ep_num):
    start_time = time.time()

    done = False

    observation = env.reset()

    env.env.frameskip = 265
    observation, reward, done, info = env.step(0)
    env.env.frameskip = (2, 5)

    processed_obs = image_preprocessor(observation)
    sum_reward = 0

    while not done:
        env.render()
        action = agent.get_action_for_test(processed_obs)
        observation, reward, done, info = env.step(action)
        agent.save_action_and_reward(action, reward)

        processed_obs = image_preprocessor(observation)

        game_step_number += 1
        sum_reward += reward

        if done:
            agent.reset_episode_related_containers()
            # agent.update()

            games_reward_list.append(sum_reward)
            games_step_num_list.append(game_step_number)
            game_step_number = 0

    # For logging purposes
    if i % 10 == 0:
        print()
        print("Number of total games: ", i+1)
        print("Total step numbers: ", agent.global_step)
        print()
        print("Mean step number: ", np.mean(games_step_num_list))
        print()
        print("Games' rewards list:")
        print(games_reward_list)
    #    print("Previously seen states number:", agent.seen_states_number)
        # print("Current Alpha is: {}".format(agent.curr_alpha()))
        print("-----------------------------------------------------------------------------")


        games_reward_list = []
        games_step_num_list = []

    # if time.time() - last_save_time > 9000:
    #     agent.full_save(save_dir)
    #     last_save_time = time.time()
    #     print("/////////////////////////////////////// SAVE ////////////////////////////////////////")
