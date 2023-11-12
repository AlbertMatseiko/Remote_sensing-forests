import h5py
import tensorflow as tf


class BatchLoader:
    def __init__(self, path_to_h5, batch_size, channels=None):
        if channels is None:
            channels = [0, 1, 2, 3, 4, 5, 6]
        self.channels = channels
        self.path = path_to_h5
        self.bs = batch_size
        with h5py.File(self.path, 'r') as f:
            self.num = f['all/data_norm'].shape[0]
        self.batch_num = self.num // self.bs

    def __call__(self):
        for i in range(self.batch_num):
            with h5py.File(self.path, 'r') as f:
                start = i * self.bs
                stop = start + self.bs
                batch = f['all/data_norm'][start:stop, :, :, self.channels]
            yield batch


def make_train_dataset(path_to_h5, batch_size, WIDTH=256, HEIGHT=256, CHANNELS_LIST=None):
    if CHANNELS_LIST is None:
        CHANNELS_LIST = [0, 1, 2, 3, 4, 5, 6]
    print(f"InMakeDataset={CHANNELS_LIST}")
    BL = BatchLoader(path_to_h5, batch_size, channels=CHANNELS_LIST)
    train_dataset = tf.data.Dataset.from_generator(
        BL,
        output_signature=(tf.TensorSpec(shape=(batch_size, WIDTH, HEIGHT, len(CHANNELS_LIST))))
    )
    train_dataset = train_dataset.repeat(-1).prefetch(tf.data.AUTOTUNE)  # хранение в логах для ускорения обучения
    return train_dataset
