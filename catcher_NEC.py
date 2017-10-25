import logging
import sys
import numpy as np
from scipy import misc

from catchertest import CatcherforTest

from nec_agent import NECAgent


def image_preprocessor(state, size=(40, 40)):
    #state = state[32:195, :, :]
    #state = misc.imresize(state, size)
    # greyscaling and normalizing state
    state = np.dot(state[..., :3], np.array([0.299, 0.587, 0.114], dtype=np.float32)) / 255.0
    return state
def setup_logging(level=logging.INFO, is_stream_handler=True, is_file_handler=False, file_handler_filename=None):
    log.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if is_stream_handler:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        log.addHandler(ch)

    if is_file_handler:
        if file_handler_filename:
            fh = logging.FileHandler(file_handler_filename)
            fh.setLevel(level)
            fh.setFormatter(formatter)
            log.addHandler(fh)
        else:
            raise ValueError("file_handler_filename must not be None if is_file_handler = True")


logging.basicConfig(stream=sys.stdout, level=logging.INFO)

nec_agent_parameters_dict = {
    "log_save_directory": "C:/Work/temp/nec_agent",
    "dnd_max_memory": 100000,
    "input_shape": (40, 40, 2),
    "fully_conn_neurons": 64,
    "neighbor_number": 25,
    "num_outputs": (16, 16, 16, 16),
    "n_step_horizon": 10,

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

env = CatcherforTest()

games_reward_list = []

for i in range(max_ep_num):
    done = False
    mini_game_done = False

    observation = env.reset()
    processed_obs = image_preprocessor(observation)

    if i % 10 == 0:
        logging.info("#### New game started. Game number: {} ####".format(i + 1))
        logging.info("Global step: {}".format(agent.global_step))

    while not done:
        action = agent.get_action(processed_obs)

        observation, reward, done = env.step(action)
        agent.save_action_and_reward(action, reward)

        processed_obs = image_preprocessor(observation)

    agent.update()
    agent.reset_episode_related_containers()

    games_reward_list.append(reward)
    if i % 10 == 0:
        print("Ucso 10 game reward összege: ", sum(games_reward_list))
        games_reward_list = []
