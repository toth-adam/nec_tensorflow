from lru import LRU
from scipy.spatial.ckdtree import cKDTree
# TODO: Use annoy instead of scipy ckdtree; profile index-building performance (we need to use index-rebuilding a lot)
# from annoy import AnnoyIndex


class DND:

    def __init__(self, num_neighbors=50, max_memory=5e5):
        self.dictionary = LRU(max_memory)
        self.num_neighbors = num_neighbors
        self.ann = None
        self.cached_embeddings = None

    # This is factored out, since the ANN implementation will change
    def _search_ann(self, lookup_key):
        # TODO: eps is for approx. neighbor; if eps=0 it gives back accurate results
        # p = 2 -- Euclidean distance
        # n_jobs = -1 -- Number of jobs to schedule for parallel processing. If -1 is given, all processors are used.
        _, indices = self.ann.query(lookup_key, k=self.num_neighbors, eps=0, p=2, n_jobs=-1)
        return [self.cached_embeddings[i] for i in indices]

    def _make_hashable_for_dict(self):
        pass

    # rebuild KDTree
    def __rebuild_kd_tree(self):
        self.cached_embeddings = self.dictionary.keys()
        self.ann = cKDTree(self.cached_embeddings, compact_nodes=True, balanced_tree=True)

    # We can define the limit as a parameter (and not the same number as the neighbor)
    def is_queryable(self):
        return True if len(self.dictionary) > self.num_neighbors else False

    def lookup(self, key):
        # key has to be a tuple -- hashable type
        assert "__hash__" in dir(key)
        # The list
        neighbor_embeddings = self._search_ann(key)
        # We modify the LRU order with this line
        neighbor_embeddings_values = [self.dictionary[embedding] for embedding in neighbor_embeddings]
        return neighbor_embeddings, neighbor_embeddings_values

    def upsert(self, key, val):
        # key has to be a tuple -- hashable type
        assert "__hash__" in dir(key)
        self.dictionary[key] = val
        # Rebuild KDTree
        self.__rebuild_kd_tree()
