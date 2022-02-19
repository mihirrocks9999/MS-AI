import tensorflow as tf
array = [1, 2, 3, 4]
arrays = [11, 22, 33, 44]

array = tf.constant(array)
arrays = tf.constant(arrays)

array = array.shuffle(4, seed=0.5)
arrays = arrays.shuffle(4, seed=0.5)