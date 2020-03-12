import os

import numpy as np

DIR = os.path.dirname(os.path.realpath(__file__))


def save_scope(scope, fpath, sess):
    saver = tf.train.Saver(
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope))
    # saver.restore(sess, fpath)
    saver.save(sess, fpath)


def load_scope(scope, fpath, sess):
    saver = tf.train.Saver(
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope))
    saver.restore(sess, fpath)


def test_sanity():
    fpath = '/tmp/test_tf_save/'
    scope = 'model'

    def _save_random_stuff():
        numpy_mda = np.random.rand(2, 3, 4)
        with tf.variable_scope(scope):
            _tf_var_from_np = tf.get_variable("my_var",
                                              initializer=numpy_mda)
        _sess = tf.Session()
        _sess.run(tf.global_variables_initializer())
        _result = _sess.run(_tf_var_from_np)
        assert np.array_equal(_result, numpy_mda)
        save_scope(scope, fpath, _sess)
        tf.reset_default_graph()
        _sess.close()
        return _result

    result = _save_random_stuff()

    with tf.variable_scope(scope):
        tf_var_from_np = tf.get_variable("my_var",
                                         initializer=np.random.rand(2, 3, 4))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    result2 = sess.run(tf_var_from_np)

    assert not all(np.equal(result, result2).flatten())

    load_scope(scope, fpath, sess)

    result3 = sess.run(tf_var_from_np)

    assert np.array_equal(result, result3)
    assert not np.array_equal(result2, result3)
    print('Tests passed!')



if __name__ == '__main__':
    test_sanity()