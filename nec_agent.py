import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from scipy import misc
from lru import LRU
from pyflann import FLANN
import time


class NECAgent:
    def __init__(self, tf_session, action_vector, dnd_max_memory=500000, neighbor_number=50):

        # ANN Search index
        self.anns = {k: AnnSearch(neighbor_number, dnd_max_memory) for k in action_vector}

        # TODO: parameters
        self.delta = 1e-3
        #
        self.initial_epsilon = 1.0

        # TODO: Átírni, hogy __init__ paraméter legyen
        # RMSProp parameters
        self.rms_learning_rate = 1e-3  # TODO: Ez gőzöm sincs hogy jó-e
        self.rms_decay = 0.9  # Állítólag DeepMind-os érték
        self.rms_epsilon = 0.01  # Állítólag DeepMind-os érték

        self.adam_learning_rate = 1e-4

        #  Tabular like update parameters
        self.tab_alpha = 1e-2

        self.action_vector = action_vector
        self.number_of_actions = len(action_vector)

        self.fully_connected_neuron = 64
        self.dnd_max_memory = int(dnd_max_memory)
        #AZ LRU az tf_index:state_hash mert az ann_search alapján kell a sorrendet updatelni mert a dict1-ben updatelni kell
        #dict1 az state_hash:tf_index ez ahhoz kell hogy megnezzem hogy benne van-e tehát milyen legyen a tab_update és hogy melyik indexre a DND-ben
        self.tf_index__state_hash = {k: LRU(self.dnd_max_memory) for k in action_vector}
        self.state_hash__tf_index = {k: {} for k in action_vector}

        # Ez követi a DND beteléséig, hogy hol állunk
        self._actual_dnd_length = {act: 0 for act in self.action_vector}

        # Tensorflow Session object
        self.session = tf_session

        # Global step
        self.global_step = 0
        self._is_search_ann_first_run = True

        # Tensorflow graph building
        # Without frame stacking
        # self.state = tf.placeholder(tf.float32, name="state")
        # With frame stacking. (84x84 mert a conv háló validja miatt nem kell hozzáfűzni a képhez)
        self.state = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.float32, name="state")

        # TODO: Át kell írni a shape-t
        self.dnd_keys = tf.Variable(
            tf.random_normal([self.number_of_actions, self.dnd_max_memory, self.fully_connected_neuron]),
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

        self.dnd_key_write = tf.scatter_nd_update(self.dnd_keys, self.dnd_write_index, [self.state_embedding])

        self.dnd_value_update = tf.placeholder(tf.float32, None, name="dnd_value_update")
        self.dnd_value_cond = tf.placeholder(tf.int32, None, name="dnd_value_condition")  # 0: hozzáad; 1: felülír

        self.dnd_value_write = tf.cond(tf.less(tf.constant(0), self.dnd_value_cond),
                                       lambda: tf.scatter_nd_update(self.dnd_values, self.dnd_write_index,
                                                                    self.dnd_value_update),
                                       lambda: tf.scatter_nd_add(self.dnd_values, self.dnd_write_index,
                                                                 self.dnd_value_update))

        self.ann_search = py_func(self._search_ann, [self.state_embedding, self.dnd_keys], tf.int32,
                                  name="ann_search", grad=_ann_gradient)
        self.nn_state_embeddings = tf.gather_nd(self.dnd_keys, self.ann_search, name="nn_state_embeddings")
        self.nn_state_values = tf.gather_nd(self.dnd_values, self.ann_search, name="nn_state_values")


        # DND calculation
        # tf.expand_dims azért kell, hogy a különböző DND kulcsokból ugyanazt kivonjuk többször (5-ös képlet)
        self.expand_dims = tf.expand_dims(tf.expand_dims(self.state_embedding, axis=1), axis=1)
        self.square_diff = tf.square(self.expand_dims - self.nn_state_embeddings)
        # Nem tudom miért kell a delta-t listába tenni, első futtatásnál kiderül majd
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

    def get_action(self, state):
        # TODO: We can use numpy.random.RandomState if we want to test the implementation

        # Mindegyik action-re megbecsüljük a Q(s_t, a)-t, ezzel fel is töltjük a DND-t
        # Az eredeti paper Algorithm #1  - 2. sora

        # Choose the random action
        if np.random.random_sample() < self.curr_epsilon():
            action = np.random.choice(self.action_vector)
        # Choose the greedy action
        else:
            max_q = self.session.run(self.predicted_q, feed_dict={self.state: state})
            action = self.action_vector[max_q[0]]

        # TODO: Have to save trajectory as well (states, state embeddings)
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

    def _search_ann(self, search_keys, dnd_keys):
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
            # Create empty array with double length
            tf_indices = np.empty([indices.shape[0], indices.shape[1] * 2], dtype=indices.dtype)
            # Összefésüljük az action indexet az annből kijövő indexszel
            tf_indices[:, 0::2] = action_indices
            tf_indices[:, 1::2] = indices
            tf_indices = tf_indices.reshape((indices.shape[0], indices.shape[1], 2))
            batch_indices.append(tf_indices)
        np_batch = np.asarray(batch_indices)

        # Olyan alakra hozzuk, ami a gather_nd tf operationhoz kell
        final_indices = np.asarray([np_batch[:, j, :, :] for j in range(np_batch.shape[1])])

        return final_indices

        # return np.array([[[[0, 0], [0, 1]], [[1, 0], [1, 1]], [[2, 0], [2, 1]]], [[[0, 0], [0, 1]], [[1, 0], [1, 1]], [[2, 0], [2, 1]]]])

    # TODO: Ezt kellene batchelve!!!
    def tabular_like_update(self, state, state_hash, action, q_n):
        action_index = self.action_vector.index(action)
        if state_hash in self.tf_index__state_hash[action]:
            cond = 0
            dnd_gather_ind = self.tf_index__state_hash[action][state_hash]
            indices = np.array([[[action_index, dnd_gather_ind]]])
            dnd_q_value = self.session.run(self.nn_state_values, feed_dict={self.ann_search: indices})
            update_value = self.tab_alpha*(q_n - dnd_q_value)

        else:
            cond = 1
            if len(self.tf_index__state_hash[action]) < self.dnd_max_memory:
                item = len(self.tf_index__state_hash[action])
                self._actual_dnd_length[action_index] = item
            else:
                _, item = self.tf_index__state_hash[action].peek_last_item()
            self.tf_index__state_hash[action][state_hash] = item
            indices = np.array([[[self.action_vector.index(action), item]]])
            update_value = q_n
            self._action_state_hash[action][item] = state_hash
            #print(indices)
            #print(update_value)

        #state_embedding = self.session.run(self.state_embedding, feed_dict={self.state: state})
        #print("state embedding:", state_embedding)
        #print("eredit kulcs: ", self.session.run(self.nn_state_embeddings, feed_dict={self.ann_search: indices}))
        self.session.run([self.dnd_value_write, self.dnd_key_write],
                         feed_dict={self.state: state,
                                    self.dnd_value_cond: cond,
                                    self.dnd_value_update: np.array([[[update_value]]]),
                                    self.dnd_write_index: indices})
        #print("update után: ", self.session.run(self.nn_state_embeddings, feed_dict={self.ann_search: indices}))


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

        tf_var_dnd_indices = [self._ann_index__tf_index[j] if j in self._ann_index__tf_index else j for index_row in indices for j in index_row]
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
    state = misc.imresize(state, [84, 84])
    # greyscaling and normalizing state
    state = np.dot(state[..., :3], [0.299, 0.587, 0.114]) / 255.0
    return state


# TODO: env_reset esetén a kapott observationt stackkelni kell 4szer : np.stack((o_t,o_t,o_t,o_t), axis=2)
def frame_stacking(s_t, o_t):  # Ahol az "s_t" a korábban stackkelt 4 frame, "o_t" pedig az új observation
    s_t1 = np.append(s_t[:, :, 1:], np.expand_dims(o_t, axis=2), axis=2)
    return s_t1


def transform_array_to_tuple(tf_array):
    return tuple(tf_array)


#if __name__ == "__main__":
#    # with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
#    with tf.Session() as sess:
#        tf.set_random_seed(1)
#        agent = NECAgent(sess, [-1, 0, 1], dnd_max_memory=1e5)
#
#        # tf.summary.FileWriter("c:\\Work\\Coding\\temp\\", graph=sess.graph)
#        #
#        # print(ops.get_gradient_function(agent.ann_search.op))
#        #
#        # print(tf.trainable_variables())
#        np.random.seed(1)
#        # fake_frame = np.random.rand(84, 84, 4)
#        # two_fake_frames = np.array([fake_frame])
#        # print(fake_frame)
#
#        # print(sess.run(agent.state_embedding, feed_dict={agent.state: fake_frame}))
#        # print(agent.test_ann_indices(fake_frame))
#        # print(agent._write_dnd(fake_frame))
#
#        # print(agent.dnd_values.eval())
#
#        # print("kaki")
#
#
#
#        #print(agent.get_action(fake_frame))
#
#        # # print(agent.get_action(fake_frame))
#        #
#        # s_e, dnd_keys, dist, w, nw, sq, pq = sess.run([agent.state_embedding, agent.nn_state_embeddings, agent.distances, agent.weightings, agent.normalised_weightings, agent.squeeze, agent.pred_q_values], feed_dict={agent.state: two_fake_frames})
#
#        # print(s_e, "\n####")
#        # print("DND__KEYS: ", dnd_keys, "\n####")
#        # print("dist", dist, "\n####")
#        # print(w, "\n####")
#        # print("norm_wei", nw, "\n####")
#        # print(agent.test_ann_indices_values(two_fake_frames))
#
#        # print("pred_q", pq, "\n####")
#
#        # print(sq,"\n K")
#
#        # print(sess.run(agent.predicted_q, feed_dict={agent.state: two_fake_frames}))
#        before = time.time()
#        # agent._write_dnd(5)
#        for _ in range(1):
#            states, actions, hashes = [], [], []
#
#            for i in range(1):
#                fake_frame = np.random.rand(84, 84, 4)
#                two_fake_frames = np.array([fake_frame])
#                states.append(two_fake_frames)
#                hashes.append(hash(two_fake_frames.tobytes()))
#                actions.append(agent.get_action(two_fake_frames))
#                if i>16 and i%16 == 0:
#                    sess.run(agent.optimizer, feed_dict={agent.state: np.array(states[:32]).reshape((32, 84, 84, 4)),
#                                                        agent.action: actions[:32],
#                                                        agent.target_q: actions[:32]})
#
#            #print(actions, hashes)
#            for s, a, h in zip(states, actions, hashes):
#                agent.tabular_like_update(s, h, a, np.random.rand())
#
#            #sess.run(agent.optimizer, feed_dict={agent.state: np.array([states]).reshape((1000,84,84,4)),
#            #                                     agent.action: actions,
#            #                                     agent.target_q: actions})
#            #agent.tabular_like_update()
#
#            # print(str(_) + ".: ", sess.run(agent.pred_q_values, feed_dict={agent.state: two_fake_frames}))
#
#        print("Idő: ", time.time() - before)

########################################################################################################################
from replay_memory import ReplayMemory
import gym
import scipy.signal
from collections import deque


def discount(x, gamma):
    a = np.asarray(x)
    return scipy.signal.lfilter([1], [1, -gamma], a[::-1], axis=0)[::-1]


session = tf.Session()
agent = NECAgent(session, [0, 2, 3], dnd_max_memory=100, neighbor_number=10)
rep_memory = ReplayMemory()
n_hor = 100
max_ep_num = 1
gamma = 0.99
gammas = list(map(lambda x: gamma ** x, range(n_hor)))
batch_size = 32

env = gym.make('Pong-v4')

print(session.run(agent.action_onehot, feed_dict={agent.action_index: [0, 2]}))

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
        action = agent.get_action(agent_input)
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
            #  BACKPROP, minden lépésnél???????
            if agent.global_step > 300:
                print("pista")
                # TODO: Az action batch átadása nem jó még itt, az action batch indexeket kell és nem a true actiont!!!!
                state_batch, action_batch, q_n_batch = rep_memory.get_batch(batch_size)
                # print(state_batch, action_batch, q_n_batch)
                session.run(agent.optimizer, feed_dict={agent.state: state_batch,
                                                        agent.action_index: action_batch,
                                                        agent.target_q: q_n_batch})

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

    print(agent.global_step)
    #print(len(rewards_deque))
    #print(rewards_deque.count(-1))
    #print(local_step)

    #TODO: Tabular like update