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