Neural Episodic Control Agent
=============================

This is an implementation of (a slightly modified version of) Deepmind's `Neural Episodic Control <https://arxiv.org/pdf/1703.01988.pdf>`_.

  The modifications:
    - The order of the LRU is modified only when the agent interacts its environment.
    - The convolutional layers are from the A3C's implementation of OpenAI. 

Dependencies
------------

- **Tensorflow**: https://github.com/tensorflow/tensorflow
- **numpy**: https://github.com/numpy/numpy
- **scipy**: https://github.com/scipy/scipy
- **LRU**: https://github.com/amitdev/lru-dict
- **FLANN**: https://github.com/Erotemic/flann
- **mmh3**: https://github.com/aappleby/smhasher
- **OpenAI gym**: https://github.com/openai/gym

Notes
-----
(In case of FLANN, a fork is used instead of the original repo, because add_points() and remove_point() methods of FLANN has no python
binding in the original repo.)
