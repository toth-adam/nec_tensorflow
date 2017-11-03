import numpy as np
from scipy import misc

import gym

from nec_agent import NECAgent, setup_logging


def image_preprocessor(state, size=(84, 84)):
    state = state[32:195, :, :]
    state = misc.imresize(state, size)
    # greyscaling and normalizing state
    state = np.dot(state[..., :3], np.array([0.299, 0.587, 0.114], dtype=np.float32)) / 255.0
    return state


setup_logging()

nec_agent_parameters_dict = {
    "log_save_directory": "C:/Work/temp/nec_agent",
    "dnd_max_memory": 100000
}

agent = NECAgent([0, 2, 3], **nec_agent_parameters_dict)

max_ep_num = 500000

env = gym.make('Pong-v4')

games_reward_list = []

for i in range(max_ep_num):
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

        if mini_game_done:
            agent.update()
            agent.reset_episode_related_containers()
