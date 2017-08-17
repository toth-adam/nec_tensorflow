import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import ops
from scipy import misc
from lru import LRU


class NECAgent:
    def __init__(self, tf_session, action_vector, dnd_max_memory=500000):

        # TODO: parameters
        self.delta = 1e-3
        #
        self.initial_epsilon = 1.0

        # TODO: Átírni, hogy __init__ paraméter legyen
        # RMSProp parameters
        self.rms_learning_rate = 1e-3  # TODO: Ez gőzöm sincs hogy jó-e
        self.rms_decay = 0.9  # Állítólag DeepMind-os érték
        self.rms_epsilon = 0.01  # Állítólag DeepMind-os érték

        #  Tabular like update parameters
        self.tab_alpha = 1e-2

        self.action_vector = action_vector
        self.number_of_actions = len(action_vector)

        self.fully_connected_neuron = 5
        self._dnd_order = {k: LRU(dnd_max_memory) for k in action_vector}

        # Tensorflow Session object
        self.session = tf_session

        # Global step
        self.global_step = 0

        # Tensorflow graph building
        # Without frame stacking
        # self.state = tf.placeholder(tf.float32, name="state")
        # With frame stacking. (84x84 mert a conv háló validja miatt nem kell hozzáfűzni a képhez)
        self.state = tf.placeholder(shape=[None, 84, 84, 2], dtype=tf.float32, name="state")

        # TODO: Át kell írni a shape-t
        self.dnd_keys = tf.Variable(tf.random_normal([self.number_of_actions, dnd_max_memory, self.fully_connected_neuron]),
                                    name="DND_keys")
        self.dnd_values = tf.Variable(tf.zeros([self.number_of_actions, dnd_max_memory, 1]), name="DND_values")

        # TODO: We have to stack exactly 4 frames now to be able to feed it into self.state (4 channel)
        self.conv1 = slim.conv2d(activation_fn=tf.nn.elu,
                                 inputs=self.state, num_outputs=32,
                                 kernel_size=[8, 8], stride=[4, 4], padding='VALID')
        self.conv2 = slim.conv2d(activation_fn=tf.nn.elu,
                                 inputs=self.conv1, num_outputs=16,
                                 kernel_size=[4, 4], stride=[2, 2], padding='VALID')
        self.conv3 = slim.conv2d(activation_fn=tf.nn.elu,
                                 inputs=self.conv2, num_outputs=16,
                                 kernel_size=[3, 3], stride=[1, 1], padding='VALID')

        # TODO: This is the final fully connected layer
        self.state_embedding = slim.fully_connected(slim.flatten(self.conv3), self.fully_connected_neuron,
                                                    activation_fn=tf.nn.elu)

        self.dnd_write_index = tf.placeholder(tf.int32, None, name="dnd_write_index")

        self.dnd_key_write = tf.scatter_nd_update(self.dnd_keys, self.dnd_write_index, self.state_embedding)

        self.dnd_value_update = tf.placeholder(tf.float32, None, name="dnd_value_update")
        self.dnd_value_cond = tf.placeholder(tf.int32, None, name="dnd_value_condition")  # 0: hozzáad; 1: felülír

        self.dnd_value_write = tf.cond(tf.less(tf.constant(0), self.dnd_value_cond),
                                       lambda: tf.scatter_nd_update(self.dnd_values, self.dnd_write_index,
                                                                    self.dnd_value_update),
                                       lambda: tf.scatter_nd_add(self.dnd_values, self.dnd_write_index,
                                                                 self.dnd_value_update))

        self.dnd_gather_index = tf.placeholder(tf.int32, None, name="dnd_gather_index")
        self.dnd_gather_value = tf.gather(self.dnd_values, self.dnd_gather_index)

        self.ann_search = py_func(self._search_ann, [self.state_embedding, self.dnd_keys], [tf.int32, tf.int32],
                                  name="ann_search", grad=_ann_gradient)

        self.nn_state_embeddings = tf.gather_nd(self.dnd_keys, self.ann_search, name="nn_state_embeddings")
        self.nn_state_values = tf.gather_nd(self.dnd_values, self.ann_search, name="nn_state_values")

        # Retrieve info from DND dictionary
        # TODO: Lehet át kell írni a 3. paramétert
        # embs_and_values = tf.py_func(self._query_dnds, [self.state_embedding, self.action], tf.float32)
        # # dnd_embeddings-eknek a 50 közeli szomszéd kulcsok kellenek
        # self.dnd_embeddings = tf.to_float(embs_and_values[0])
        # self.dnd_values = tf.to_float(embs_and_values[1])


        # DND calculation
        # tf.expand_dims azért kell, hogy a különböző DND kulcsokból ugyanazt kivonjuk többször (5-ös képlet)
        square_diff = tf.square(self.nn_state_embeddings - tf.expand_dims(self.state_embedding, 1))
        # Nem tudom miért kell a delta-t listába tenni, első futtatásnál kiderül majd
        distances = tf.reduce_sum(square_diff, axis=2) + [self.delta]
        weightings = 1.0 / distances
        # A normalised_weightings a 2-es képlet
        normalised_weightings = weightings / tf.reduce_sum(weightings, axis=1, keep_dims=True)
        # Ez az 1-es képlet
        self.pred_q_values = tf.reduce_sum(self.nn_state_values * normalised_weightings, axis=1,
                                           name="predicted_q_values")
        self.predicted_q = tf.argmax(self.pred_q_values, name="predicted_q")

        # Ennek egy vektornak kell lennie. pl: [1, 0, 0]
        self.action = tf.placeholder(tf.int32, [None], name="action")
        self.action_onehot = tf.one_hot(self.action, self.number_of_actions)

        # Loss Function
        # TODO: Ez miért stringes float?? miért nem tf.float
        self.target_q = tf.placeholder(tf.float32, [None], name="target_Q")
        self.q_value = tf.reduce_sum(tf.multiply(self.pred_q_values, self.action_onehot))
        self.td_err = tf.subtract(self.target_q, self.q_value, name="td_error")
        total_loss = tf.reduce_sum(tf.square(self.td_err, name="squared_error"), name="total_loss")

        # Optimizer
        # self.optimizer = tf.train.RMSPropOptimizer(self.rms_learning_rate, decay=self.rms_decay,
        #                                            epsilon=self.rms_epsilon).minimize(total_loss)
        self.optimizer = tf.train.AdamOptimizer(self.rms_learning_rate).minimize(total_loss)

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
        # processed_state = image_preprocessor(state)
        processed_state = state

        pred_q_values = self.session.run(self.pred_q_values, feed_dict={self.state: processed_state})

        # Choose the random action
        if np.random.random_sample() < curr_epsilon():  # TODO: step_nr nincs inicializálva!!!
            action = np.random.choice(self.action_vector)
        # Choose the greedy action
        else:
            action = self.action_vector[np.argmax(pred_q_values)]

        # TODO: Have to save trajectory as well (states, state embeddings)
        self.global_step += 1
        return action

    def test_ann_indices(self, state):
        return self.session.run(self.nn_state_embeddings, feed_dict={self.state: state})

    def get_state_embeddings(self, state):
        return self.session.run(self.state_embedding, feed_dict={self.state: state})

    def get_embeddings_and_q_values(self, state, actions):
        # This returns a len=2 list: 1st: nd-array representing the state-embedding
        #                            2nd: len=len(actions) list with the Q values for the actions
        return self.session.run([self.state_embedding, self.pred_q_values], feed_dict={self.state: state,
                                                                                       self.action: actions})

    def reset(self):
        pass

    def curr_epsilon(self):
        eps = self.initial_epsilon
        if 4999 < self.global_step < 25000:
            eps = self.initial_epsilon - ((self.global_step - 5000) * 4.995e-5)
        elif self.global_step > 24999:
            eps = 0.001
        return eps

    def _search_ann(self, search_key, dnd_keys):
        return [[[0, 1], [1, 0]], [[0, 0], [1, 1]]]
        # return [[[0, 0], [0, 2], [1, 0], [1, 2], [2, 0], [2, 2]]]

    def tabular_like_update(self, state_hash, action, q_n):
        if state_hash in self._dnd_order[action]:
            dnd_gather_ind = self._dnd_order[action][state_hash] # itt a visszakapott indexet olyanna kell tenni hogy a tf.gather beszopkodja
            gather_indices = [[[[dnd_gather_ind, 1, action]]]]  # ha mar atirodik a dnd shape, meg ennek a shapje is lehet hogy szar
            dnd_q_value = self.session.run(self.dnd_gather_value, feed_dict={self.dnd_gather_index: gather_indices})
            update_value = self.tab_alpha*(q_n - dnd_q_value)
            self.session.run(self.dnd_value_write,
                             feed_dict={self.dnd_value_cond: [0],
                                        self.dnd_value_update: [update_value],
                                        self.dnd_write_index: gather_indices})
            write_indices = [[[action, dnd_gather_ind]]]  # itt viszont jó ha nem az action az ucsó dimenzió
            # a square_diff esetében is lefuttatjuk a state_embeddinget, van duplikáció?
            self.session.run(self.dnd_key_write, feed_dict={self.dnd_write_index: write_indices})
        else:
            last_item = self._dnd_order[action].peek_last_item()
            del self._dnd_order[action][last_item[0]]
            self._dnd_order[action][state_hash] = last_item[1]
            self.session.run(self.dnd_value_write,
                             feed_dict={self.dnd_value_cond: [1],
                                        self.dnd_value_update:[q_n],
                                        self.dnd_write_index:[[[[last_item[1], 1, action]]]]})
            write_indices = [[[action, last_item[1]]]] #itt viszont jó ha nem az action az ucsó dimenzió
            self.session.run(self.dnd_key_write, feed_dict={self.dnd_write_index: write_indices})
            #AZÉRT ÁT KELL MAJD NÉZNI MERT LEHET ELKURESZOLTAM VALAMIT

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


if __name__ == "__main__":
    with tf.Session() as sess:
        agent = NECAgent(sess, [-1, 0, 1], dnd_max_memory=10)

        # tf.summary.FileWriter("c:\\Work\\Coding\\temp\\", graph=sess.graph)
        #
        # print(ops.get_gradient_function(agent.ann_search.op))
        #
        # print(tf.trainable_variables())
        fake_frame = np.random.rand(1, 84, 84, 2)

        # print(sess.run(agent.state_embedding, feed_dict={agent.state: fake_frame}))

        # print(agent._write_dnd(fake_frame))

        print(agent.dnd_keys.eval())

        print("kaki")

        print(agent.test_ann_indices(fake_frame))

        # print(agent.get_action(fake_frame))
