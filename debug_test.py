import numpy as np

from nec_agent import NECAgent


def image_preprocessor(state):
    return state


nec_agent_parameters_dict = {
    "log_save_directory": "C:/Work/temp/nec_agent",
    "dnd_max_memory": 10,
    "neighbor_number": 2,
    "input_shape": (5, 5, 2),
    "batch_size": 2,
    "n_step_horizon": 3,
    "epsilon_decay_bounds": (10, 25),
    "optimization_start": 10
}

agent = NECAgent([0, 2, 3], **nec_agent_parameters_dict)

max_ep_num = 10

games_reward_list = []

np.random.seed(0)


def step(a):
    return np.full((5, 5), i+0.1*j, np.float32), i+0.1*j, False if j <= 5 else True, {}


for i in range(max_ep_num):
    j = 0
    done = False
    mini_game_done = False

    observation = np.full((5, 5), i, np.float32)
    processed_obs = image_preprocessor(observation)

    while not done:
        j += 1
        action = agent.get_action(processed_obs)

        observation, reward, done, info = step(action)
        agent.save_action_and_reward(action, reward)

        processed_obs = image_preprocessor(observation)

        if done:
            agent.update()
            agent.reset_episode_related_containers()
