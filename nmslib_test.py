import nmslib
import numpy as np
import time


# create a random matrix to index
# data = np.zeros((20, 10), dtype=np.float32) + np.arange(10, 20, dtype=np.float32)
# data_t = np.array(data.T, dtype=np.float32, copy=True)

np.random.seed(1)

data = np.random.rand(10, 20).astype(np.float32)
data_t = data

print(data)
print(data_t)

query = np.ones(20, dtype=np.float32)

# initialize a new index, using a HNSW index on Cosine Similarity
indexing_time1 = time.time()
index = nmslib.init(method='hnsw', space='l2')
index.addDataPointBatch(data_t)
# index.createIndex({'post': 2}, print_progress=True)
index.createIndex({'post': 2})
indexing_time2 = time.time()

# query for the nearest neighbours of the first datapoint
# ids, distances = index.knnQuery(data[0], k=10)

# get all nearest neighbours for all the datapoint
# using a pool of 4 threads to compute
# neighbours = index.knnQueryBatch(data, k=10, num_threads=4)

ids, distances = index.knnQuery(query, k=3)

print(ids, distances)

# t1 = time.time()
#
# neighbours = index.knnQueryBatch(data[:10000], k=50, num_threads=4)
#
# print("Index creation: ", indexing_time2 - indexing_time1, "[s]")
# print("Query: ", time.time() - t1, "[s]")
#
#
# data_2 = numpy.random.randn(100, 100).astype(numpy.float32)
#
# index.addDataPointBatch(data_2)
# index.createIndex({'post': 2}, print_progress=True)
#
# ids, distances = index.knnQuery(data_2[0], k=10)

print(data_t[4])
print(np.linalg.norm(data_t[4] - query))
print(np.sqrt(np.sum((data_t[4] - query) **2)))