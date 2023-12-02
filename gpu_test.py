import tensorflow as tf

tf.debugging.set_log_device_placement(True)
tf.ones([])
# [...] op Fill in device /job:localhost/replica:0/task:0/device:GPU:0
with tf.device("CPU"):
    tf.ones([])
