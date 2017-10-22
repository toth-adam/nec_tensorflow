import logging
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

log = logging.getLogger(__name__)
setup_logging(log)

nec_agent_parameters_dict = {
    "log_save_directory": "C:/Work/temp/nec_agent",
    "dnd_max_memory": 100000
}

agent = NECAgent([0, 2, 3], **nec_agent_parameters_dict)

# LOADING
# if True:
#     load_path = "C:/RL/nec_saves"
#     agent.load_agent(load_path, 2750)
#     rep_memory.load(load_path, 2750)
#     for action_index, act in enumerate(agent.action_vector):
#         dnd_keys = session.run(agent.dnd_keys)
#         agent.anns[act].build_index(dnd_keys[action_index][:agent._dnd_length(act)])

max_ep_num = 500000

env = gym.make('Pong-v4')

games_reward_list = []

for i in range(max_ep_num):
    done = False
    mini_game_done = False

    observation = env.reset()
    processed_obs = image_preprocessor(observation)

    log.info("#### Pong new game started. (21 points.) Game number: {} ####".format(i + 1))

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

    #mini_games_reward_sum = sum(mini_game_rewards_list)
    #games_reward_list.append(mini_games_reward_sum)
    #log.info("Mini-game step numbers: {}".format(local_step_list))
    #log.info("Mini-game rewards: {}".format(mini_game_rewards_list))
#
    #log.info("Score for a (21-points) game: {}".format(mini_games_reward_sum))
    #for act, dnd in agent.tf_index__state_hash.items():
    #    log.info("DND length for action {}: {}".format(act, len(dnd)))
    #log.info("Global step number: {}".format(agent.global_step))
#
    #if (i + 1) % 10 == 0 and i != 0:
    #    log.info("Score average for last 10 (21-points) game: {}".format(sum(games_reward_list[-10:]) / 10))
#
    #if (i + 1) % 5 == 0 and i != 0:
    #    save_path = "C:/Work/temp/nec_agent"
    #    agent.save_agent(save_path)
    #    rep_memory.save(save_path, agent.global_step)
