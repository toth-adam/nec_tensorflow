import numpy as np
import tensorflow as tf
from dnd import DND


class NECAgent:

    def __init__(self, tf_session, action_vector):

        # TODO: parameters
        self.delta = 1e-3
        # Initial epsilon-greedy policy parameter # HYPERPARAMETER
        # Quote from the paper: "In practice, we use epsilon-greedy policy during training with a low epsilon"
        #
        self.initial_epsilon = 0.7

        # TODO: Átírni, hogy __init__ paraméter legyen
        # RMSProp parameters
        self.rms_learning_rate = 1e-3  # TODO: Ez gőzöm sincs hogy jó-e
        self.rms_decay = 0.9  # Állítólag DeepMind-os érték
        self.rms_epsilon = 0.01  # Állítólag DeepMind-os érték

        self.action_vector = action_vector
        self.number_of_actions = len(action_vector)
        self.dnds = {k: DND() for k in action_vector}

        # Tensorflow Session object
        self.session = tf_session

        # Tensorflow graph building
        self.state = tf.placeholder(tf.float32, )
        self.action = tf.placeholder(tf.int8, [None])

        # TODO: ConvNet part of TFGraph

        # TODO: This is the final fully connected layer?????
        self.state_embedding = ...

        # Retrieve info from DND dictionary
        # TODO: Lehet át kell írni a 3. paramétert
        embs_and_values = tf.py_func(self._query_dnds, [self.state_embedding, self.action], tf.float32)
        # dnd_embeddings-eknek a 50 közeli szomszéd kulcsok kellenek
        self.dnd_embeddings = tf.to_float(embs_and_values[0])
        self.dnd_values = tf.to_float(embs_and_values[1])

        # DND calculation
        # tf.expand_dims azért kell, hogy a különböző DND kulcsokból ugyanazt kivonjuk többször (5-ös képlet)
        square_diff = tf.square(self.dnd_embeddings - tf.expand_dims(self.state_embedding, 1))
        # Nem tudom miért kell a delta-t listába tenni, első futtatásnál kiderül majd
        distances = tf.reduce_sum(square_diff, axis=2) + [self.delta]
        weightings = 1.0 / distances
        # A normalised_weightings a 2-es képlet
        normalised_weightings = weightings / tf.reduce_sum(weightings, axis=1, keep_dims=True)
        # Ez az 1-es képlet
        self.pred_q = tf.reduce_sum(self.dnd_values * normalised_weightings, axis=1)

        # Loss Function
        # TODO: Ez miért stringes float?? miért nem tf.float
        self.target_q = tf.placeholder(tf.float32, [None])
        self.td_err = self.target_q - self.pred_q
        total_loss = tf.reduce_sum(tf.square(self.td_err))

        # Optimizer
        self.optimizer = tf.train.RMSPropOptimizer(self.rms_learning_rate, decay=self.rms_decay,
                                                   epsilon=self.rms_epsilon).minimize(total_loss)

    def get_action(self, state):
        # TODO: We can use numpy.random.RandomState if we want to test the implementation
        # Choose the random action
        if np.random.random_sample() < self._current_epsilon():
            action = np.random.choice(self.action_vector)
        # Choose the greedy action
        else:
            processed_state = image_preprocessor(state)
            embedding = self.get_state_embeddings(processed_state)

            dnd_values = [self._query_dnds(embedding, action) for action in self.action_vector]
            # q_values = self.

    def get_state_embeddings(self, state):
        return self.session.run(self.state_embedding, feed_dict={self.state: state})

    def get_q_values(self, embedding, action):
        return self.session.run(self.pred_q, feed_dict={self.state_embedding: embedding, self.action: action})

    def reset(self):
        pass

    # def get_state(self):
    #     pass

    # TODO: Implement this
    def _current_epsilon(self, decay=0):
        return self.initial_epsilon - decay * self.initial_epsilon

    # TODO: Két külön függvényt kell írni. Egyik ami a tf.py_func-ba megy, a másik meg ami a self.get_action-be
    def _query_dnds(self, state_embedding, action):
        action_specific_dnd = self.dnds.get(action)
        # If we are at the start of the training
        if action_specific_dnd.is_queryable():
            # Transform to tuple
            # TODO: Maybe move it outside of this function?
            state_embedding_tuple = transform_array_to_tuple(state_embedding)
            return action_specific_dnd.lookup(state_embedding_tuple)
        else:
            pass


def image_preprocessor(state):
    # Implement, if necessary
    return state


def transform_array_to_tuple(tf_array):
    return tuple(tf_array)
