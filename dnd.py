from lru import LRU
from sklearn.neighbors import LSHForest
# TODO: Use annoy instead of scikit-learn; profile index-building performance (we need to use index-rebuilding a lot)
# from annoy import AnnoyIndex


class DND:

    def __init__(self, num_neighbors=50, max_memory=5e5):
        self.dictionary = LRU(max_memory)
        self.ann = LSHForest(n_neighbors=num_neighbors)

    # This is factored out, since the ANN implementation will change
    def _search_ann(self, lookup_key):

        return None

    # LSHForest specific stuff
    def __update_lsh_forest(self, key):
        self.ann.partial_fit(key)

    def lookup(self, key):
        # The list
        neighbor_embeddings = self._search_ann(key)
        neighbor_embeddings_values =
        # TODO: Vissza kéne adnia num_neighbors szerinti kulcsokat és értékeket (anélkül h. szétbasznánk az LRU sorrendet)
        return neighbor_embeddings, neighbor_embeddings_values

    def upsert(self, key, val):
        self.dictionary[key] = val
