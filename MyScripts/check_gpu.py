import tensorflow as tf
print(tf.config.list_physical_devices('GPU'), tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    print(gpu)
    tf.config.experimental.set_memory_growth(gpu, True)
