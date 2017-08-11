"""
Kérdés 1.: Ha van ugyanolyan hash értékű elem a dictionaryban, akkor a state_embedding_key értékét az újra cserélem?
(ha esetleg a régi az más, mert ugye a conv háló paraméterei frissülnek közben)

"""
from lru import LRU
from scipy.spatial.ckdtree import cKDTree
import tensorflow as tf
# TODO: Use annoy instead of scipy ckdtree; profile index-building performance (we need to use index-rebuilding a lot)
# TODO: TF Variable valahogy a DND-be. (Egy tenzor??) hogy lehet darabjait update-elni? VAgy ha csak azt az ötvenet adom
# TODO: vissza, akkor mivel csak azok befolyásolják a loss-t, csak azok updatelődnek?
# from annoy import AnnoyIndex


class DND:

    def __init__(self, num_neighbors=50, max_memory=5e5):
        self.state_embeddings = tf.Variable(tf.zeros([max_memory, 256], dtype=tf.float64), name="dnd_keys")
        self.state_embeddings_values = tf.Variable(tf.zeros([max_memory, 1]), name="dnd_values")

        self.state_hash_dictionary = LRU(int(max_memory))
        self.num_neighbors = num_neighbors
        self.ann = None
        self.cached_embeddings = None

        self.

    def __len__(self):
        return len(self.dictionary)

    # This is factored out, since the ANN implementation will change
    def _search_ann(self, lookup_key):
        # TODO: eps is for approx. neighbor; if eps=0 it gives back accurate results
        # p = 2 -- Euclidean distance
        # n_jobs = -1 -- Number of jobs to schedule for parallel processing. If -1 is given, all processors are used.
        _, indices = self.ann.query(lookup_key, k=self.num_neighbors, eps=0, p=2, n_jobs=-1)
        return [self.cached_embeddings[i] for i in indices]

    # rebuild KDTree
    def __rebuild_kd_tree(self):
        self.cached_embeddings = [embedding for embedding, _ in self.dictionary.values()]
        self.ann = cKDTree(self.cached_embeddings, compact_nodes=True, balanced_tree=True)

    # We can define the limit as a parameter (and not the same number as the neighbor)
    def is_queryable(self):
        return True if len(self) > self.num_neighbors else False

    def is_state_in_dictionary(self, original_state_hash):
        return original_state_hash in self.dictionary

    def get_values(self, original_state_hash):
        # Returns state_embedding and associated value
        return self.dictionary[original_state_hash]

    def lookup_neighbors(self, embedding_key):
        # The list
        neighbor_embeddings = self._search_ann(embedding_key)
        # We modify the LRU order with this line
        neighbor_embeddings_values = [self.dictionary[original_state_hash] for embedding in neighbor_embeddings]
        return neighbor_embeddings, neighbor_embeddings_values

    def upsert(self, key, val):
        # key = hash on the original state
        # val = [state_embedding_tf_tensor, value] -- It has to be a list, so its updateable

        # Note: Must be run on 64bit, since on 32bit system python __hash__ function generates a 32 bit hash
        # with that, the hash collision probability is almost certain (0.9999999999997707)
        # with 64 bit: 6.776250005557927e-09
        assert len(val) == 2
        self.dictionary[key] = val
        # Rebuild KDTree
        self.__rebuild_kd_tree()

    def upsert_tf(self):

