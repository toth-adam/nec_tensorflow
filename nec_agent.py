import tensorflow as tf
from dnd import DND


class NECAgent:

    def __init__(self, action_vector):

        # TODO: parameters
        self.delta = 1e-3

        # TODO: Átírni, hogy __init__ paraméter legyen
        # RMSProp parameters
        self.rms_learning_rate = 1e-3  # TODO: Ez gőzöm sincs hogy jó-e
        self.rms_decay = 0.9  # Állítólag DeepMind-os érték
        self.rms_epsilon = 0.01  # Állítólag DeepMind-os érték

        self.action_vector = action_vector
        self.number_of_actions = len(action_vector)
        self.dnds = {k: DND() for k in action_vector}

        self.action = tf.placeholder(tf.int8, [None])

        # TODO: ConvNet part of TFGraph

        # TODO: This is the final fully connected layer?????
        self.state_embedding = ...

        # Retrieve info from DND dictionary
        embs_and_values = tf.py_func(self._query_dnds, [self.state_embedding, self.action], [tf.float64, tf.float64])
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
        self.target_q = tf.placeholder("float", [None])
        self.td_err = self.target_q - self.pred_q
        total_loss = tf.reduce_sum(tf.square(self.td_err))

        # Optimizer
        self.optimizer = tf.train.RMSPropOptimizer(self.rms_learning_rate, decay=self.rms_decay,
                                                   epsilon=self.rms_epsilon).minimize(total_loss)

    def _query_dnds(self, state_embedding, action):
        action_specific_dnd = self.dnds.get(action)
        return action_specific_dnd.lookup(state_embedding)
