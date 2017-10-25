import logging
import sys
import os
from collections import deque

import numpy as np
from scipy.signal import lfilter

import tensorflow as tf
import tensorflow.contrib.slim as slim

from lru import LRU
from pyflann import FLANN

from replay_memory import ReplayMemory


log = logging.getLogger(__name__)


class NECAgent:
    def __init__(self, action_vector, cpu_only=False, dnd_max_memory=500000, neighbor_number=50,
                 backprop_learning_rate=1e-4, tabular_learning_rate=0.5e-2, fully_conn_neurons=128,
                 input_shape=(84, 84, 4), kernel_size=((3, 3), (3, 3), (3, 3), (3, 3)), num_outputs=(32, 32, 32, 32),
                 stride=((2, 2), (2, 2), (2, 2), (2, 2)), delta=1e-3, rep_memory_size=1e5, batch_size=32,
                 n_step_horizon=100, discount_factor=0.99, log_save_directory=None, epsilon_decay_bounds=(5000, 25000),
                 optimization_start=1000):

        self._cpu_only = cpu_only

        # ----------- HYPERPARAMETERS ----------- #

        self.delta = delta
        self.initial_epsilon = 1
        self.epsilon_decay_bounds = epsilon_decay_bounds

        # Optimizer parameters
        self.adam_learning_rate = backprop_learning_rate
        self.batch_size = batch_size
        self.optimization_start = optimization_start

        # Tabular parameters
        self.tab_alpha = tabular_learning_rate
        self.dnd_max_memory = int(dnd_max_memory)

        # Reinforcement learning parameters
        self.n_step_horizon = n_step_horizon
        self.discount_factor = discount_factor

        # Convolutional layer parameters
        self._input_shape = input_shape
        self.fully_connected_neuron = fully_conn_neurons
        self._kernel_size = kernel_size
        self._stride = stride
        self._num_outputs = num_outputs

        # Environment specific parameters
        self.action_vector = action_vector
        self.number_of_actions = len(action_vector)
        self.frame_stacking_number = input_shape[-1]

        # ANN Search index
        self.anns = {k: AnnSearch(neighbor_number, dnd_max_memory, k) for k in action_vector}

        # Replay memory
        self.replay_memory = ReplayMemory(size=rep_memory_size, stack_size=input_shape[-1])

        #AZ LRU az tf_index:state_hash mert az ann_search alapján kell a sorrendet updatelni mert a dict1-ben
        # updatelni kell dict1 az state_hash:tf_index ez ahhoz kell hogy megnezzem hogy benne van-e tehát milyen
        # legyen a tab_update és hogy melyik indexre a DND-ben
        self.tf_index__state_hash = {k: LRU(self.dnd_max_memory) for k in action_vector}
        self.state_hash__tf_index = {k: {} for k in action_vector}

        # Tensorflow Session object
        self.session = self._create_tf_session(self._cpu_only)

        # Step numbers
        self.global_step = 0
        self.episode_step = 0
        self.episode_number = 0

        # ----------- TENSORFLOW GRAPH BUILDING ----------- #

        self.state = tf.placeholder(shape=[None, *self._input_shape], dtype=tf.float32, name="state")

        # TF Variables representing the Differentiable Neural Dictionary (DND)
        self.dnd_keys = tf.Variable(
            tf.random_normal([self.number_of_actions, self.dnd_max_memory, self.fully_connected_neuron]),
            name="DND_keys")
        self.dnd_values = tf.Variable(tf.random_normal([self.number_of_actions, self.dnd_max_memory, 1]),
                                      name="DND_values")

        # Always better to use smaller kernel size! These layers are from OpenAI
        # Learning Atari: An Exploration of the A3C Reinforcement
        # TODO: USE 1x1 kernels-bottleneck, CS231n Winter 2016: Lecture 11 from 29 minutes
        self.convolutional_layers = self._create_conv_layers()

        # This is the final fully connected layer
        self.state_embedding = slim.fully_connected(slim.flatten(self.convolutional_layers[-1]),
                                                    self.fully_connected_neuron, activation_fn=tf.nn.elu)

        # DND write operations
        self.dnd_write_index = tf.placeholder(tf.int32, None, name="dnd_write_index")

        self.dnd_key_write = tf.scatter_nd_update(self.dnd_keys, self.dnd_write_index, self.state_embedding)

        self.dnd_value_update = tf.placeholder(tf.float32, None, name="dnd_value_update")

        self.dnd_value_write = tf.scatter_nd_update(self.dnd_values, self.dnd_write_index, self.dnd_value_update)

        # This placeholder is used to decide whether modify LRU order in the DND or not (We modify the order during
        # action selection for new frames; we do not modify the order if we run the optimizer.)
        self.is_update_LRU_order = tf.placeholder(tf.int32, None, name="is_LRU_order_update")
        # Custom function to handle Approximate Nearest Neighbor search
        self.ann_search = py_func(self._search_ann, [self.state_embedding, self.dnd_keys, self.is_update_LRU_order],
                                  tf.int32, name="ann_search", grad=_ann_gradient)

        # Gather operations to select from DND (according to ann search outputs)
        self.nn_state_embeddings = tf.gather_nd(self.dnd_keys, self.ann_search, name="nn_state_embeddings")
        self.nn_state_values = tf.gather_nd(self.dnd_values, self.ann_search, name="nn_state_values")

        # DND calculation
        # expand_dims() is needed to subtract the key(s) (state_embedding) from neighboring keys (Eq. 5)
        self.expand_dims = tf.expand_dims(tf.expand_dims(self.state_embedding, axis=1), axis=1)
        self.square_diff = tf.square(self.expand_dims - self.nn_state_embeddings)

        # We clip the values here, because the 0 values cause problems during backward pass (NaNs)
        self.distances = tf.sqrt(tf.clip_by_value(tf.reduce_sum(self.square_diff, axis=3), 1e-12, 1e12)) + self.delta
        self.weightings = 1.0 / self.distances
        # Normalised weightings (Eq. 2)
        self.normalised_weightings = self.weightings / tf.reduce_sum(self.weightings, axis=2, keep_dims=True)
        # (Eq. 1)
        self.squeeze = tf.squeeze(self.nn_state_values, axis=3)
        self.pred_q_values = tf.reduce_sum(self.squeeze * self.normalised_weightings, axis=2,
                                           name="predicted_q_values")
        self.predicted_q = tf.argmax(self.pred_q_values, axis=1, name="predicted_q")

        # This has to be an iterable, e.g.: [1, 0, 0]
        self.action_index = tf.placeholder(tf.int32, [None], name="action")
        self.action_onehot = tf.one_hot(self.action_index, self.number_of_actions, axis=-1)

        # Loss Function
        self.target_q = tf.placeholder(tf.float32, [None], name="target_Q")
        self.q_value = tf.reduce_sum(tf.multiply(self.pred_q_values, self.action_onehot), axis=1)
        self.td_err = tf.subtract(self.target_q, self.q_value, name="td_error")
        total_loss = tf.square(self.td_err, name="total_loss")

        # Optimizer
        self.optimizer = tf.train.AdamOptimizer(self.adam_learning_rate).minimize(total_loss)

        # ----------- AUXILIARY ----------- #
        # ----------- TF related ----------- #

        # Global initialization
        self.init_op = tf.global_variables_initializer()
        self.session.run(self.init_op)

        # Check op for NaN checking - if needed
        self.check_op = tf.add_check_numerics_ops()

        # Saver op
        self.saver = tf.train.Saver(max_to_keep=5)

        # ----------- Episode related containers ----------- #
        self._observation_list = []
        self._agent_input_list = []
        self._agent_input_hashes_list = []
        self._agent_action_list = []
        self._rewards_deque = deque()
        self._q_values_list = []

        # Logging
        self._log_hyperparameters()

        # Create discount factor vector
        self._gammas = list(map(lambda x: self.discount_factor ** x, range(self.n_step_horizon)))

        # Create epsilon decay rate (Now it is linearly decreasing between 1 and 0.001)
        self._epsilon_decay_rate = (1 - 0.001) / (self.epsilon_decay_bounds[1] - self.epsilon_decay_bounds[0])

    # This is the main function which we call in different environments during playing
    def get_action(self, processed_observation):
        # Get the agent input (frame-stacking) using the preprocessed observation
        # We also store the relevant quantities
        agent_input = self._get_agent_input(processed_observation)
        # Get the action
        action = self._get_action(agent_input)
        # Optimize if the global_step number is above optimization_start (making sure we have enough elements in the
        # replay memory and each DND)
        if self.global_step > self.optimization_start:
            self._optimize()
            # Calculate bootstrap Q value as early as possible, so we can insert the corresponding (S, A, Q) tuple into
            # the replay memory. Because of this, the agent may sample from this example during the next _optimize()
            # call. (Intentionally)
            if len(self._rewards_deque) == self.n_step_horizon:
                q = self._calculate_bootstrapped_q_value()
                # Store (S, A, Q) in the replay memory
                self._add_to_replay_memory(q)
                # We pop the leftmost element from the rewards deque, hence the condition before
                # _calculate_bootstrapped_q_value() remains True until the episode end.
                # (Also we do not need this element anymore, since we have already used it for calculating the Q value.)
                self._rewards_deque.popleft()

        self.global_step += 1
        self.episode_step += 1

        return action

    # This is the main function which we call in different environments after an episode is finished.
    def update(self):
        #  játék vége van kiszámolom a disc_rewardokat viszont az elsőnek n_hor darab rewardból
        #  a másodiknak (n_hor-1) darab rewardból, a harmadiknak (n_hor-2) darab rewardból, ésígytovább.
        #  A bootstrap value itt mindig 0 tehát a Q(N) maga a discounted reward. Majd berakosgatom a replay memoryba
        # Itt van lekezelve az, hogy a játék elején Monte-Carlo return-nel számoljuk ki a state-action value-kat.

        q_ns = self._discount(self._rewards_deque)
        self._add_to_replay_memory_episode_end(q_ns)
        index_rebuild = not bool(self.episode_number % 10)

        # TODO: Index rebuild bool here  -- index_rebuild = not bool(mini_game_counter % 10)

        self._tabular_like_update(self._agent_input_list, self._agent_input_hashes_list,
                                  self._agent_action_list, self._q_values_list, index_rebuild)

    def reset_episode_related_containers(self):
        self._observation_list = []
        self._agent_input_list = []
        self._agent_input_hashes_list = []
        self._agent_action_list = []
        self._rewards_deque = deque()
        self._q_values_list = []
        self.episode_step = 0
        # Increment episode number
        self.episode_number += 1

    # Should be a pre-processed observation
    def _get_agent_input(self, processed_observation):
        if self.episode_step == 0:
            agent_input = self._initial_frame_stacking(processed_observation)
        else:
            agent_input = self._frame_stacking(self._agent_input_list[-1], processed_observation)
        # Saving the relevant quantities
        self._observation_list.append(processed_observation)
        self._agent_input_list.append(agent_input)
        print(hash(agent_input.tobytes()))
        self._agent_input_hashes_list.append(hash(agent_input.tobytes()))
        return agent_input

    def _get_action(self, agent_input):
        # Choose the random action
        if np.random.random_sample() < self.curr_epsilon():
            action = np.random.choice(self.action_vector)
        # Choose the greedy action
        else:
            # We expand the agent_input dimensions here to run the graph for batch_size = 1 -- action selection
            max_q = self.session.run(self.predicted_q, feed_dict={self.state: np.expand_dims(agent_input, axis=0),
                                                                  self.is_update_LRU_order: 1})
            log.debug("Max. Q value: {}".format(max_q[0]))
            action = self.action_vector[max_q[0]]
            log.debug("Chosen action: {}".format(action))

        return action

    def _optimize(self):
        # Get the batches from replay memory and run optimizer
        state_batch, action_batch, q_n_batch = self.replay_memory.get_batch(self.batch_size)
        action_batch_indices = [self.action_vector.index(a) for a in action_batch]
        self.session.run(self.optimizer, feed_dict={self.state: state_batch,
                                                    self.action_index: action_batch_indices,
                                                    self.target_q: q_n_batch,
                                                    self.is_update_LRU_order: 0})
        log.debug("Optimizer has been run.")

    def _add_to_replay_memory(self, q, episode_end=False):
        s = self._observation_list[self.episode_step - self.n_step_horizon]
        a = self._agent_action_list[self.episode_step - self.n_step_horizon]
        self.replay_memory.append((s, a, q), episode_end)

    def _add_to_replay_memory_episode_end(self, q_list):
        j = len(self._rewards_deque)
        for i, (o, a, q_n) in enumerate(zip(self._observation_list[-j:], self._agent_action_list[-j:], q_list)):
            self._q_values_list.append(q_n)
            e_e = False
            if i == j - 1:
                e_e = True
            self.replay_memory.append([o, a, q_n], e_e)

    # Note that this function calculate only one Q at a time.
    def _calculate_bootstrapped_q_value(self):
        discounted_reward = np.dot(self._rewards_deque, self._gammas)
        bootstrap_value = np.amax(self.session.run(self.pred_q_values,
                                                   feed_dict={self.state: [self._agent_input_list[self.episode_step]],
                                                              self.is_update_LRU_order: 0}))
        disc_bootstrap_value = self.discount_factor ** self.n_step_horizon * bootstrap_value
        q_value = discounted_reward + disc_bootstrap_value

        # Store calculated Q value
        self._q_values_list.append(q_value)
        return q_value

    def _calculate_q_values_at_episode_end(self):
        pass

    def _insert_into_replay_memory(self, state, action, q):
        self.replay_memory.append()

    def curr_epsilon(self):
        eps = self.initial_epsilon
        if self.epsilon_decay_bounds[0] <= self.global_step < self.epsilon_decay_bounds[1]:
            eps = self.initial_epsilon - ((self.global_step - self.epsilon_decay_bounds[0]) * self._epsilon_decay_rate)
        elif self.global_step >= self.epsilon_decay_bounds[1]:
            eps = 0.001
        return eps

    def _search_ann(self, search_keys, dnd_keys, update_LRU_order):
        batch_indices = []
        for act, ann in self.anns.items():
            # These are the indices we get back from ANN search
            indices = ann.query(search_keys)
            log.debug("ANN indices for action {}: {}".format(act, indices))
            # Create numpy array with full of corresponding action vector index
            action_indices = np.full(indices.shape, self.action_vector.index(act))
            log.debug("Action indices for action {}: {}".format(act, action_indices))
            # Riffle two arrays
            tf_indices = self._riffle_arrays(action_indices, indices)
            batch_indices.append(tf_indices)
            # Very important part: Modify LRU Order here
            # Doesn't work without tabular update of course!
            if update_LRU_order == 1:
                _ = [self.tf_index__state_hash[act][i] for i in indices.ravel()]
        np_batch = np.asarray(batch_indices)
        log.debug("Batch update indices: {}".format(np_batch))

        # Reshaping to gather_nd compatible format
        final_indices = np.asarray([np_batch[:, j, :, :] for j in range(np_batch.shape[1])], dtype=np.int32)

        return final_indices

    def _tabular_like_update(self, states, state_hashes, actions, q_ns, index_rebuild):
        log.debug("Tabular like update has been started.")
        # Making np arrays
        states = np.asarray(states)
        state_hashes = np.asarray(state_hashes)
        q_ns = np.asarray(q_ns)
        actions = np.asarray(actions)

        action_indices = np.asarray([self.action_vector.index(act) for act in actions])

        # DND Lengths before modification
        dnd_lengths = self._dnd_lengths()

        dnd_q_values = np.empty(q_ns.shape, dtype=np.float32)
        dnd_gather_indices = np.asarray([self.state_hash__tf_index[a][sh] if sh in self.state_hash__tf_index[a]
                                         else None for sh, a in zip(state_hashes, actions)])

        in_cond_vector = dnd_gather_indices != None
        indices = np.squeeze(self._riffle_arrays(action_indices[in_cond_vector], dnd_gather_indices[in_cond_vector]),
                             axis=0)

        dnd_q_vals = self.session.run(self.nn_state_values, feed_dict={self.ann_search: indices})
        dnd_q_vals = np.squeeze(dnd_q_vals, axis=1)
        dnd_q_values[in_cond_vector] = dnd_q_vals

        local_sh_dict = {a: {} for a in self.action_vector}

        # Batch means one complete game (21-points) in this context
        batch_update_values = []
        batch_indices = []
        batch_states = []
        batch_indices_for_ann = []
        batch_valid_indices = np.full(q_ns.shape, False, dtype=np.bool)
        batch_cond_vector = []
        ii = 0

        for j, (act, sh, q, state) in enumerate(zip(actions, state_hashes, q_ns, states)):
            if sh in self.state_hash__tf_index[act] and sh not in local_sh_dict[act]:
                update_value = self.tab_alpha * (q - dnd_q_values[j]) + dnd_q_values[j]
                local_sh_dict[act][sh] = (ii, update_value)

                # Add elements to lists
                batch_states.append(state)
                batch_indices.append(dnd_gather_indices[j])
                batch_update_values.append(update_value)
                batch_indices_for_ann.append(dnd_gather_indices[j])
                batch_valid_indices[j] = True
                # ANN related - Append True because it is already added to ANN points
                batch_cond_vector.append(True)
                ii += 1

            elif sh in self.state_hash__tf_index[act] and sh in local_sh_dict[act]:
                # We are not adding elements to the lists in this case
                update_value = self.tab_alpha * (q - local_sh_dict[act][sh][1]) + local_sh_dict[act][sh][1]
                ind = local_sh_dict[act][sh][0]
                batch_update_values[ind] = update_value
                local_sh_dict[act][sh] = (ind, update_value)
            else:
                if len(self.tf_index__state_hash[act]) < self.dnd_max_memory:
                    index = len(self.tf_index__state_hash[act])
                else:
                    index, old_state_hash = self.tf_index__state_hash[act].peek_last_item()
                    del self.state_hash__tf_index[act][old_state_hash]
                # LRU order stuff
                self.tf_index__state_hash[act][index] = sh
                self.state_hash__tf_index[act][sh] = index

                # Add elements to lists and update local_sh_dict
                local_sh_dict[act][sh] = (ii, q)

                batch_states.append(state)
                batch_indices.append(index)
                batch_update_values.append(q)
                batch_indices_for_ann.append(index)
                batch_valid_indices[j] = True
                batch_cond_vector.append(False)
                ii += 1

        batch_states = np.asarray(batch_states, dtype=np.float32)
        batch_indices = np.asarray(batch_indices, dtype=np.int32)
        batch_update_values = np.asarray(batch_update_values, dtype=np.float32)
        batch_indices_for_ann = np.asarray(batch_indices_for_ann, dtype=np.int32)
        batch_cond_vector = np.asarray(batch_cond_vector, dtype=np.bool)

        # Create batch indices and update values for TensorFlow session
        batch_indices = np.squeeze(self._riffle_arrays(action_indices[batch_valid_indices], batch_indices))
        batch_update_values = np.expand_dims(batch_update_values, axis=1)

        # Batch tabular update
        state_embeddings, _, _ = self.session.run([self.state_embedding, self.dnd_value_write, self.dnd_key_write],
                                                  feed_dict={self.state: batch_states,
                                                  self.dnd_value_update: batch_update_values,
                                                  self.dnd_write_index: batch_indices})

        # FLANN Add point - every batch
        if not index_rebuild:
            for a in self.action_vector:
                act_cond = actions[batch_valid_indices] == a
                self.anns[a].update_ann(batch_indices_for_ann[act_cond], state_embeddings[act_cond],
                                        batch_cond_vector[act_cond], dnd_lengths[self.action_vector.index(a)])

        # FLANN index rebuild, if index_rebuild = True
        if index_rebuild:
            dnd_keys = self.session.run(self.dnd_keys)
            for act, ann in self.anns.items():
                action_index = self.action_vector.index(act)
                # Ez a jó (kövi sor)
                ann.build_index(dnd_keys[action_index][:self._dnd_length(act)])

        log.debug("Tabular like update has been run.")

    def save_action_and_reward(self, a, r):
        self._agent_action_list.append(a)
        self._rewards_deque.append(r)

    def _save_q_value(self, q):
        self._q_values_list.append(q)

    def _discount(self, x):
        a = np.asarray(x)
        return lfilter([1], [1, -self.discount_factor], a[::-1], axis=0)[::-1]

    def _dnd_lengths(self):
        return [len(self.tf_index__state_hash[a]) for a in self.action_vector]

    def _dnd_length(self, a):
        return len(self.tf_index__state_hash[a])

    def _create_conv_layers(self):
        """
        Create convolutional layers in the Tensorflow graph according to the hyperparameters, using Tensorflow slim
        library.

        Returns
        -------
        conv_layers: list
            The list of convolutional operations.

        """
        lengths_set = {len(o) for o in (self._num_outputs, self._kernel_size, self._stride)}
        if len(lengths_set) != 1:
            msg = "The lengths of the conv. layers params vector should be same. Lengths: {}, Vectors: {}".format(
                [len(o) for o in (self._num_outputs, self._kernel_size, self._stride)],
                (self._num_outputs, self._kernel_size, self._stride))
            raise ValueError(msg)
        conv_layers = []
        inputs = [self.state]
        for i, (num_out, kernel, stride) in enumerate(zip(self._num_outputs, self._kernel_size, self._stride)):
            layer = slim.conv2d(activation_fn=tf.nn.elu, inputs=inputs[i], num_outputs=num_out,
                                kernel_size=kernel, stride=stride, padding='SAME')
            conv_layers.append(layer)
            inputs.append(layer)
        return conv_layers

    @staticmethod
    def _riffle_arrays(array_1, array_2):
        if len(array_1.shape) == 1:
            array_1 = np.expand_dims(array_1, axis=0)
            array_2 = np.expand_dims(array_2, axis=0)

        tf_indices = np.empty([array_1.shape[0], array_1.shape[1] * 2], dtype=array_1.dtype)
        # Riffle the action indices with ann output indices
        tf_indices[:, 0::2] = array_1
        tf_indices[:, 1::2] = array_2
        return tf_indices.reshape((array_1.shape[0], array_1.shape[1], 2))

    def _initial_frame_stacking(self, processed_obs):
        return np.stack((processed_obs, ) * self.frame_stacking_number, axis=2)

    @staticmethod
    def _frame_stacking(s_t, o_t):  # Ahol az "s_t" a korábban stackkelt 4 frame, "o_t" pedig az új observation
        s_t1 = np.append(s_t[:, :, 1:], np.expand_dims(o_t, axis=2), axis=2)
        return s_t1

    def save_agent(self, path):
        self.saver.save(self.session, path + '/model_' + str(self.global_step) + '.cptk')
        # az LRU mappán belül hozza létre az actionokhöz tartozó .npy fájlt.
        # Ebből létre lehet hozni a "self.state_hash__tf_index" is!
        try:
            os.mkdir(path + '/LRU_' + str(self.global_step))
        except FileExistsError:
            pass
        for a, dict in self.tf_index__state_hash.items():
            np.save(path + '/LRU_' + str(self.global_step) + "/" + str(a) + '.npy', dict.items())

        # TODO: Save replay memory here

    def load_agent(self, path, glob_step_num):
        self.saver.restore(self.session, path + "/model_" + str(glob_step_num) + '.cptk')
        self.global_step = glob_step_num
        for a in self.action_vector:
            act_LRU = np.load(path + '/LRU_' + str(glob_step_num) + "/" + str(a) + '.npy')
            # azért reversed, hogy a lista legelső elemét rakja bele utoljára, így az lesz az MRU
            for tf_index, state_hash in reversed(act_LRU):
                self.tf_index__state_hash[a][tf_index] = state_hash
                self.state_hash__tf_index[a][state_hash] = tf_index

    def _log_hyperparameters(self):
        pass

    @staticmethod
    def _create_tf_session(only_cpu):
        if only_cpu:
            config = tf.ConfigProto(device_count={"GPU": 0})
            return tf.Session(config=config)
        else:
            return tf.Session()


class AnnSearch:

    def __init__(self, neighbors_number, dnd_max_memory, action):
        self.ann = FLANN()
        self.neighbors_number = neighbors_number
        self._ann_index__tf_index = {}
        self.dnd_max_memory = int(dnd_max_memory)
        self._removed_points = 0
        self.flann_params = None
        # For logging purposes
        self.action = action

    def add_state_embedding(self, state_embedding):
        self.ann.add_points(state_embedding)

    def update_ann(self, tf_var_dnd_indices, state_embeddings, cond_vector, dnd_actual_length):
        # A tf_var_dnd_index alapján kell törölnünk a Flann indexéből. Ez csak abban az esetben fog
        # kelleni, ha nincs index build és egy olyan index jön be, amihez tartozó state_embeddeinget már egyszer hozzáadtam.

        # Ha láttuk már a pontot akkor ki kell törölni, mert a state hash-hehz tartozó state embedding érték megváltozott
        # és azt tároljuk ANN-ben
        flann_indices_seen = []
        for tf_var_dnd_index in tf_var_dnd_indices[cond_vector]:
            if tf_var_dnd_index in self._ann_index__tf_index.values():
                index = [k for k, v in self._ann_index__tf_index.items() if v == tf_var_dnd_index][0]
            else:
                index = tf_var_dnd_index
            flann_indices_seen.append(index)
        # flann_indices_seen = [k for k, v in self._ann_index__tf_index.items() if v in tf_var_dnd_indices[cond_vector]]
        self.ann.remove_points(flann_indices_seen)

        for i, tf_var_dnd_index in enumerate(tf_var_dnd_indices[cond_vector]):
            self._ann_index__tf_index[dnd_actual_length + self._removed_points + i] = tf_var_dnd_index

        # Itt adjuk hozzá a FLANN indexéhez a már látott state hash-hez
        if len(state_embeddings[cond_vector]) != 0:
            self.add_state_embedding(state_embeddings[cond_vector])

        self._removed_points += len(flann_indices_seen)

        # Ha nem láttuk és tele vagyunk
        # debug2_list = []
        counter = 0
        #print(len(tf_var_dnd_indices[~cond_vector]))
        for i, tf_var_dnd_index in enumerate(tf_var_dnd_indices[~cond_vector]):
            if dnd_actual_length + i >= self.dnd_max_memory:
                if tf_var_dnd_index in self._ann_index__tf_index.values():
                    index = [k for k, v in self._ann_index__tf_index.items() if v == tf_var_dnd_index][0]
                else:
                    index = tf_var_dnd_index

                # ez a rész itt még zsivány, nem fölfele
                self.ann.remove_point(index)
                self._ann_index__tf_index[dnd_actual_length + self._removed_points + counter] = tf_var_dnd_index

                self._removed_points += 1

            else:
                self._ann_index__tf_index[dnd_actual_length + self._removed_points + counter] = tf_var_dnd_index

                counter += 1

        self.add_state_embedding(state_embeddings[~cond_vector])

    def build_index(self, tf_variable_dnd):
        self.flann_params = self.ann.build_index(tf_variable_dnd, algorithm="kdtree", target_precision=1)
        self._ann_index__tf_index = {}
        self._removed_points = 0
        # log.info("ANN index has been rebuilt for action {}.".format(self.action))

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
