import numpy as np
import tensorflow as tf
import os

def next_batch():
    datasets = np.asarray(range(0, 20))
    input_queue = tf.train.slice_input_producer([datasets], shuffle=False)
    data_batchs = tf.train.batch(input_queue, batch_size=6, num_threads=1,
                                 capacity=18, allow_smaller_final_batch=False)
    return data_batchs


if __name__ == "__main__":
    data_batchs = next_batch()
    sess = tf.Session()
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)
    try:
        while not coord.should_stop():
            data = sess.run([data_batchs])
            print(data)
    except tf.errors.OutOfRangeError:
        print("complete")
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()
