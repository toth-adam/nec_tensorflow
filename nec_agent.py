import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from scipy import misc
from replay_memory import ReplayMemory
import gym
from scipy.signal import lfilter
from collections import deque
from lru import LRU
from pyflann import FLANN
import time


class NECAgent:
    def __init__(self, tf_session, action_vector, dnd_max_memory=500000, neighbor_number=50):

        # Hyperparameters
        self.delta = 1e-3
        self.initial_epsilon = 1.0

        # RMSProp parameters
        #self.rms_learning_rate = 1e-3
        #self.rms_decay = 0.9
        #self.rms_epsilon = 0.01

        # ADAM parameters
        self.adam_learning_rate = 1e-4

        #  Tabular like update parameters
        self.tab_alpha = 1e-2

        self.action_vector = action_vector
        self.number_of_actions = len(action_vector)

        self.fully_connected_neuron = 128
        self.dnd_max_memory = int(dnd_max_memory)

        # ANN Search index
        self.anns = {k: AnnSearch(neighbor_number, dnd_max_memory) for k in action_vector}

        #AZ LRU az tf_index:state_hash mert az ann_search alapján kell a sorrendet updatelni mert a dict1-ben
        # updatelni kell dict1 az state_hash:tf_index ez ahhoz kell hogy megnezzem hogy benne van-e tehát milyen
        # legyen a tab_update és hogy melyik indexre a DND-ben
        self.tf_index__state_hash = {k: LRU(self.dnd_max_memory) for k in action_vector}
        self.state_hash__tf_index = {k: {} for k in action_vector}

        # Ez követi a DND beteléséig, hogy hol állunk
        self._actual_dnd_length = {act: 0 for act in self.action_vector}

        # Tensorflow Session object
        self.session = tf_session

        # Global step
        self.global_step = 0
        # Első indexbuild
        self._is_search_ann_first_run = True

        # Tensorflow graph building

        # With frame stacking. (84x84 mert a conv háló validja miatt nem kell hozzáfűzni a képhez)
        self.state = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.float32, name="state")

        # TODO: Helyesebb lenne figyelni hogy mennyire van betelve a DND
        self.dnd_keys = tf.Variable(
            tf.random_normal([self.number_of_actions, self.dnd_max_memory, self.fully_connected_neuron], mean=100000),
            name="DND_keys")
        self.dnd_values = tf.Variable(tf.random_normal([self.number_of_actions, self.dnd_max_memory, 1]),
                                      name="DND_values")

        # Always better to use smaller kernel size! These layers are from OpenAI
        # Learning Atari: An Exploration of the A3C Reinforcement
        # TODO: USE 1x1 kernels-bottleneck, CS231n Winter 2016: Lecture 11 from 29 minutes

        self.conv1 = slim.conv2d(activation_fn=tf.nn.elu,
                                 inputs=self.state, num_outputs=32,
                                 kernel_size=[3, 3], stride=[2, 2], padding='SAME')
        self.conv2 = slim.conv2d(activation_fn=tf.nn.elu,
                                 inputs=self.conv1, num_outputs=32,
                                 kernel_size=[3, 3], stride=[2, 2], padding='SAME')
        self.conv3 = slim.conv2d(activation_fn=tf.nn.elu,
                                 inputs=self.conv2, num_outputs=32,
                                 kernel_size=[3, 3], stride=[2, 2], padding='SAME')
        self.conv4 = slim.conv2d(activation_fn=tf.nn.elu,
                                 inputs=self.conv3, num_outputs=32,
                                 kernel_size=[3, 3], stride=[2, 2], padding='SAME')

        # TODO: This is the final fully connected layer
        self.state_embedding = slim.fully_connected(slim.flatten(self.conv4), self.fully_connected_neuron,
                                                    activation_fn=tf.nn.elu)

        self.dnd_write_index = tf.placeholder(tf.int32, None, name="dnd_write_index")

        self.dnd_key_write = tf.scatter_nd_update(self.dnd_keys, self.dnd_write_index, self.state_embedding)

        self.dnd_value_update = tf.placeholder(tf.float32, None, name="dnd_value_update")

        self.dnd_value_write = tf.scatter_nd_update(self.dnd_values, self.dnd_write_index, self.dnd_value_update)

        self.is_update_LRU_order = tf.placeholder(tf.int32, None, name="is_LRU_order_update")
        self.ann_search = py_func(self._search_ann, [self.state_embedding, self.dnd_keys, self.is_update_LRU_order],
                                  tf.int32, name="ann_search", grad=_ann_gradient)

        self.nn_state_embeddings = tf.gather_nd(self.dnd_keys, self.ann_search, name="nn_state_embeddings")
        self.nn_state_values = tf.gather_nd(self.dnd_values, self.ann_search, name="nn_state_values")


        # DND calculation
        # tf.expand_dims azért kell, hogy a különböző DND kulcsokból ugyanazt kivonjuk többször (5-ös képlet)
        self.expand_dims = tf.expand_dims(tf.expand_dims(self.state_embedding, axis=1), axis=1)
        self.square_diff = tf.square(self.expand_dims - self.nn_state_embeddings)

        self.distances = tf.sqrt(tf.reduce_sum(self.square_diff, axis=3)) + self.delta
        self.weightings = 1.0 / self.distances
        # A normalised_weightings a 2-es képlet
        self.normalised_weightings = self.weightings / tf.reduce_sum(self.weightings, axis=2, keep_dims=True)
        # Ez az 1-es képlet
        self.squeeze = tf.squeeze(self.nn_state_values, axis=3)
        self.pred_q_values = tf.reduce_sum(self.squeeze * self.normalised_weightings, axis=2,
                                           name="predicted_q_values")
        self.predicted_q = tf.argmax(self.pred_q_values, axis=1, name="predicted_q")

        # Ennek egy vektornak kell lennie. pl: [1, 0, 0]
        self.action_index = tf.placeholder(tf.int32, [None], name="action")
        self.action_onehot = tf.one_hot(self.action_index, self.number_of_actions, axis=-1)

        # Loss Function
        self.target_q = tf.placeholder(tf.float32, [None], name="target_Q")
        self.q_value = tf.reduce_sum(tf.multiply(self.pred_q_values, self.action_onehot), axis=1)
        self.td_err = tf.subtract(self.target_q, self.q_value, name="td_error")
        total_loss = tf.square(self.td_err, name="total_loss")

        # Optimizer
        # self.optimizer = tf.train.RMSPropOptimizer(self.rms_learning_rate, decay=self.rms_decay,
        #                                            epsilon=self.rms_epsilon).minimize(total_loss)
        self.optimizer = tf.train.AdamOptimizer(self.adam_learning_rate).minimize(total_loss)

        # Global initialization
        self.init_op = tf.global_variables_initializer()

        self.session.run(self.init_op)

    def _write_dnd(self, state):

        self.session.run(self.dnd_value_write, feed_dict={self.dnd_write_index: [[[0, 0], [1, 0], [2, 0]]],
                                                                 self.dnd_value_cond: 0,
                                                                 self.dnd_value_update: [[[9], [3], [5.2]]]})

    def get_action(self, state, is_up_LRU_ord):

        # Choose the random action
        if np.random.random_sample() < self.curr_epsilon():
            action = np.random.choice(self.action_vector)
        # Choose the greedy action
        else:
            max_q = self.session.run(self.predicted_q, feed_dict={self.state: state,
                                                                  self.is_update_LRU_order: is_up_LRU_ord})
            action = self.action_vector[max_q[0]]

        self.global_step += 1
        return action

    def test_ann_indices(self, state):
        return self.session.run(self.nn_state_embeddings, feed_dict={self.state: state})

    def test_ann_indices_values(self, state):
        return self.session.run(self.nn_state_values, feed_dict={self.state: state})

    def reset(self):
        pass

    def curr_epsilon(self):
        eps = self.initial_epsilon
        if 4999 < self.global_step < 25000:
            eps = self.initial_epsilon - ((self.global_step - 5000) * 4.995e-5)
        elif self.global_step > 24999:
            eps = 0.001
        return eps

    def _search_ann(self, search_keys, dnd_keys, update_LRU_order):
        if self._is_search_ann_first_run:
            for i, ann in self.anns.items():
                ann.build_index(dnd_keys[self.action_vector.index(i)])
            self._is_search_ann_first_run = False
            print("lefutottam ti kis gecik")

        # Ezt át kell írni batches-re
        batch_indices = []
        for act, ann in self.anns.items():
            # These are the indices we get back from ANN search
            indices = ann.query(search_keys)
            # Create numpy array with full of corresponding action vector index
            action_indices = np.full(indices.shape, self.action_vector.index(act))
            # Riffle two arrays
            tf_indices = self._riffle_arrays(action_indices, indices)
            batch_indices.append(tf_indices)
            # Very important part: Modify LRU Order here
            # Doesn't work without tabular update of course!
            # TODO: ennek még utána kell nézni - A ReplyaMemoryból vett cuccokra is változtatja a sorrendet -> nem jó
            if update_LRU_order == 1:
                _ = [self.tf_index__state_hash[act][i] for i in indices.ravel()]
        np_batch = np.asarray(batch_indices)

        # Olyan alakra hozzuk, ami a gather_nd tf operationhoz kell
        final_indices = np.asarray([np_batch[:, j, :, :] for j in range(np_batch.shape[1])], dtype=np.int32)

        return final_indices

        # return np.array([[[[0, 0], [0, 1]], [[1, 0], [1, 1]], [[2, 0], [2, 1]]], [[[0, 0], [0, 1]], [[1, 0], [1, 1]],
        # [[2, 0], [2, 1]]]])

    @staticmethod
    def _riffle_arrays(array_1, array_2):
        if array_1.shape != array_2.shape:
            raise ValueError("foscsi")
        if len(array_1.shape) == 1:
            array_1 = np.expand_dims(array_1, axis=0)
            array_2 = np.expand_dims(array_2, axis=0)

        tf_indices = np.empty([array_1.shape[0], array_1.shape[1] * 2], dtype=array_1.dtype)
        # Összefésüljük az action indexet az annből kijövő indexszel
        tf_indices[:, 0::2] = array_1
        tf_indices[:, 1::2] = array_2
        return tf_indices.reshape((array_1.shape[0], array_1.shape[1], 2))

    def tabular_like_update(self, states, state_hashes, actions, q_ns, index_rebuild=False):
        # Making np arrays
        states = np.asarray(states)
        state_hashes = np.asarray(state_hashes)
        q_ns = np.asarray(q_ns)
        actions = np.asarray(actions)

        action_indices = np.asarray([self.action_vector.index(act) for act in actions])

        cond_vector = np.asarray([True if st_h in self.state_hash__tf_index[a] else False
                                  for st_h, a in zip(state_hashes, actions)])

        batch_update_values = np.empty(q_ns.shape, dtype=np.float32)
        batch_indices = np.empty((actions.shape[0], 2), dtype=np.int32)

        if np.any(cond_vector):
            dnd_gather_indices = np.asarray([self.state_hash__tf_index[a][sh]
                                            for sh, a in zip(state_hashes[cond_vector], actions[cond_vector])])
            indices = np.squeeze(self._riffle_arrays(action_indices[cond_vector], dnd_gather_indices), axis=0)
            dnd_q_values = self.session.run(self.nn_state_values, feed_dict={self.ann_search: indices})
            #print(q_ns[cond_vector], "\n#####",dnd_q_values)
            batch_update_values[cond_vector] = self.tab_alpha * (q_ns[cond_vector] - dnd_q_values) + dnd_q_values
            batch_indices[cond_vector] = indices

        sh_not_in_indices = []
        for act, sh in zip(actions[~cond_vector], state_hashes[~cond_vector]):
            if len(self.tf_index__state_hash[act]) < self.dnd_max_memory:
                index = len(self.tf_index__state_hash[act])
                # self._actual_dnd_length[action_index] = item
            else:
                index, old_state_hash = self.tf_index__state_hash[act].peek_last_item()
                del self.state_hash__tf_index[act][old_state_hash]
            # LRU order stuff
            self.tf_index__state_hash[act][index] = sh
            self.state_hash__tf_index[act][sh] = index
            #
            sh_not_in_indices.append(index)

        # Create batch indices and update values
        batch_indices[~cond_vector] = np.squeeze(self._riffle_arrays(action_indices[~cond_vector],
                                                                     np.asarray(sh_not_in_indices)))
        batch_update_values[~cond_vector] = q_ns[~cond_vector]
        batch_update_values = np.expand_dims(batch_update_values, axis=1)

        # Batch tabular update
        state_embeddings, _, _ = self.session.run([self.state_embedding, self.dnd_value_write, self.dnd_key_write],
                                                  feed_dict={self.state: states,
                                                  self.dnd_value_update: batch_update_values,
                                                  self.dnd_write_index: batch_indices})
        # TODO: Ez még kell
        # FLANN Add point - every batch  -- Szét kell szedni minden state embeddinget action csoportokba
        #for s_e, a in zip(state_embeddings, actions):
        #   self.anns[a].update_ann(, s_e)

        # FLANN index rebuild, if index_rebuild = True
        if index_rebuild:
            dnd_keys = self.session.run(self.dnd_keys)
            for act, ann in self.anns.items():
                ann.build_index(dnd_keys[self.action_vector.index(act)])


class AnnSearch:

    def __init__(self, neighbors_number, dnd_max_memory):
        self.ann = FLANN()
        self.neighbors_number = neighbors_number
        self._ann_index__tf_index = {}
        self.dnd_max_memory = int(dnd_max_memory) - 1
        self._added_points = 0
        self.flann_params = None

    def add_state_embedding(self, state_embedding):
        self.ann.add_points(state_embedding)

    def update_ann(self, tf_var_dnd_index, state_embedding):
        # A tf_var_dnd_index alapján kell törölnünk a Flann indexéből. Ez csak abban az esetben fog
        # kelleni, ha nincs index build és egy olyan index jön be, amihez tartozó state_embeddeinget már egyszer hozzáadtam.
        flann_index = [k for k, v in self._ann_index__tf_index.items() if v == tf_var_dnd_index][0]
        self.ann.remove_point(flann_index)
        self.add_state_embedding(state_embedding)
        self._added_points += 1
        self._ann_index__tf_index[self.dnd_max_memory + self._added_points] = tf_var_dnd_index

    def build_index(self, tf_variable_dnd):
        self.flann_params = self.ann.build_index(tf_variable_dnd, algorithm="kdtree", target_precision=1)
        self._ann_index__tf_index = {}

    def query(self, state_embeddings):
        indices, _ = self.ann.nn_index(state_embeddings, num_neighbors=self.neighbors_number,
                                       checks=self.flann_params["checks"])

        tf_var_dnd_indices = [[self._ann_index__tf_index[j] if j in self._ann_index__tf_index else j for j in index_row]
                              for index_row in indices]
        return np.asarray(tf_var_dnd_indices, dtype=np.int32)


def _ann_gradient(op, grad):
    return grad


# Define custom py_func which takes also a grad op as argument:
def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1000000))

    tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)


def image_preprocessor(state):
    state = state[32:195, :, :]
    state = misc.imresize(state, [84, 84])
    # greyscaling and normalizing state
    state = np.dot(state[..., :3], [0.299, 0.587, 0.114]) / 255.0
    return state


def frame_stacking(s_t, o_t):  # Ahol az "s_t" a korábban stackkelt 4 frame, "o_t" pedig az új observation
    s_t1 = np.append(s_t[:, :, 1:], np.expand_dims(o_t, axis=2), axis=2)
    return s_t1


def transform_array_to_tuple(tf_array):
    return tuple(tf_array)


def discount(x, gamma):
    a = np.asarray(x)
    return lfilter([1], [1, -gamma], a[::-1], axis=0)[::-1]

if __name__ == "__main_":
    session = tf.Session()
    env = gym.make('Pong-v4')
    agent = NECAgent(session, [0, 2, 3], dnd_max_memory=1e5, neighbor_number=50)

    observation = env.reset()
    for _ in range(30):
        observation,_,_,_=env.step(0)
    processed_obs = image_preprocessor(observation)

    import matplotlib.pyplot as plt

    plt.imshow(processed_obs, cmap="gray")
    plt.show()
    #dnd_q_value, full_dnd = session.run([agent.nn_state_values, agent.dnd_values], feed_dict={agent.ann_search: [[0, 1], [1,0]]})
    #print(dnd_q_value)
    ##print("#############x")
    ##print(full_dnd)
    #fake_frame = np.random.rand(2, 84, 84, 4)
#
    #session.run([agent.dnd_value_write, agent.dnd_key_write],
    #                 feed_dict={agent.state: fake_frame,
    #                            agent.dnd_value_update: np.asarray([[1], [2]]),
    #                            agent.dnd_write_index: np.asarray([[0, 1], [1, 0]])})
    #print(session.run([agent.dnd_values, agent.dnd_keys]))

if __name__ == "__main__":
    session = tf.Session()
    agent = NECAgent(session, [0, 2, 3], dnd_max_memory=1e5, neighbor_number=50)
    rep_memory = ReplayMemory()
    n_hor = 100
    max_ep_num = 500
    gamma = 0.99
    gammas = list(map(lambda x: gamma ** x, range(n_hor)))
    batch_size = 32

    env = gym.make('Pong-v4')

    for i in range(max_ep_num):
        rewards_deque = deque()
        states_list = []
        states_hashes_list = []
        actions_list = []
        q_n_list = []
        done = False
        mini_game_done = False
        local_step = 0

        observation = env.reset()
        processed_obs = image_preprocessor(observation)
        agent_input = np.stack((processed_obs, processed_obs, processed_obs, processed_obs), axis=2)
        states_list.append(agent_input)
        states_hashes_list.append(hash(agent_input.tobytes()))
        # Ezt is hozzá adódik a replay memoryhoz

        while not done or not mini_game_done:
            action = agent.get_action(agent_input, 1)
            actions_list.append(action)

            observation, reward, done, info = env.step(action)
            rewards_deque.append(reward)
            local_step += 1

            # mini_game_done változó beállítása, mert a gym env 21 pong játékot vesz egy játéknak
            mini_game_done = True if abs(reward) == 1 else False

            if not mini_game_done:
                #  képet megfelelőre alakítom, stackelem majd appendelem és a hash-t is appendelem
                processed_obs = image_preprocessor(observation)
                agent_input = frame_stacking(agent_input, processed_obs)
                states_list.append(agent_input)
                states_hashes_list.append(hash(agent_input.tobytes()))

                if agent.global_step > 800:
                    # TODO: Az action batch átadása nem jó még itt, az action batch indexeket kell és nem a true actiont!!!!
                    state_batch, action_batch, q_n_batch = rep_memory.get_batch(batch_size)
                    action_batch_indices = [agent.action_vector.index(a) for a in action_batch]
                    # print(state_batch, action_batch, q_n_batch)
                    session.run(agent.optimizer, feed_dict={agent.state: state_batch,
                                                            agent.action_index: action_batch_indices,
                                                            agent.target_q: q_n_batch,
                                                            agent.is_update_LRU_order: 0})

                    #  nincs játék vége még és már volt n_hor-nyi lépés akkor kiszámolom a Q(N) értéket és hozzáadom
                    #  a megfelelő vektort a replay memoryhoz
                    # Ez azért fog működni, mert a végén pop-olunk és a reward deque hossza fix marad
                    if len(rewards_deque) == n_hor:
                        disc_reward = np.dot(rewards_deque, gammas)
                        bootstrap_value = gamma ** n_hor * np.amax(session.run(agent.pred_q_values,
                                                                               feed_dict={agent.state: [states_list[local_step]]}))
                        q_n = disc_reward + bootstrap_value
                        q_n_list.append(q_n)
                        rep_memory.append([states_list[local_step - n_hor], actions_list[local_step - n_hor], q_n])
                        rewards_deque.popleft()

            else:
                #  játék vége van kiszámolom a disc_rewardokat viszont az elsőnek n_hor darab rewardból
                #  a másodiknak (n_hor-1) darab rewardból, a harmadiknak (n_hor-2) darab rewardból, ésígytovább.
                #  A bootstrap value itt mindig 0 tehát a Q(N) maga a discounted reward. Majd berakosgatom a replay memoryba

                # Itt van lekezelve az, hogy a játék elején Monte-Carlo return-nel számoljuk ki a state-action value-kat.
                q_ns = discount(rewards_deque, gamma)
                j = len(rewards_deque)
                for s, a, q_n in zip(states_list[-j:], actions_list[-j:], q_ns):
                    q_n_list.append(q_n)
                    rep_memory.append([s, a, q_n])

                # Tabular like update and ANN index rebuild
                index_rebuild = not bool(i % 10)
                agent.tabular_like_update(states_list, states_hashes_list, actions_list, q_n_list,
                                          index_rebuild=index_rebuild)

                print("i:", i, "rewardsum:", sum(rewards_deque)) if index_rebuild else None

                rewards_deque = deque()
                states_list = []
                states_hashes_list = []
                actions_list = []
                q_n_list = []
                local_step = 0

                #  képet megfelelőre alakítom, stackelem majd appendelem és a hash-t is appendelem
                processed_obs = image_preprocessor(observation)
                agent_input = np.stack((processed_obs, processed_obs, processed_obs, processed_obs), axis=2)
                states_list.append(agent_input)
                states_hashes_list.append(hash(agent_input.tobytes()))
