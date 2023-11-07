import h5py
from datetime import datetime
import os

# importing tensorflow, check gpu
import tensorflow as tf

tfl = tf.keras.layers

print(tf.config.list_physical_devices('GPU'), tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    print(gpu)
    tf.config.experimental.set_memory_growth(gpu, True)

# importing from local scripts
from TrainingNN.DataLoad import make_train_dataset
from TrainingNN.BuildNN import make_resnet_model
from TrainingNN.Visualize import VisualClass

path_to_h5 = './DATA/h5_files/LC08_L2SP_02_T1_256.h5'
with h5py.File(path_to_h5, 'r') as f:
    NUM_OF_PICTURES = len(f['all/data_norm'])
    print('Number of small images in h5:', NUM_OF_PICTURES)

# Global variables, do not change
WIDTH = 256
HEIGHT = 256
CHANNELS = 7
CLASSES = 14
MAX_SHIFT = 1  # максимальное смещение по вертикали и горизонтали в функции потерь
BATCH_SIZE = 4
DEPTH = 3
CONTRAST_FACTOR = 0.03
MAX_DELTA = 0.03
CUTTING = 1  # set to contract the step per epoch in CUTTING times
APPLY_CONV_LOSS = True
APPLY_SHIFTS = APPLY_CONV_LOSS

LR = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.03,
                                                    decay_steps=NUM_OF_PICTURES // BATCH_SIZE,
                                                    decay_rate=0.98)

# Hyperparameters
filters = [128, 64, 64]
conv_kernel = [2, 2, 2]

model, model_name, classificator = make_resnet_model(filters, conv_kernel, depth=DEPTH,
                                                     apply_conv_loss=APPLY_CONV_LOSS, apply_shifts=APPLY_SHIFTS,
                                                     CHANNELS=CHANNELS, CLASSES=CLASSES, WIDTH=WIDTH, HEIGHT=HEIGHT,
                                                     BATCH_SIZE=BATCH_SIZE,
                                                     CONTRAST_FACTOR=CONTRAST_FACTOR, MAX_DELTA=MAX_DELTA)
print(model_name)

# making dir for model if necessary
os.makedirs('./models/' + model_name, exist_ok=True)
# make a dir for tensorboard logs
logdir = "./models/logs_tb/" + model_name + "/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(logdir)
print('directory for tb logs is created')


# Make callbacks: draw a pic after every epoch, early stopping, model checkpoint, logs to tensorboard
class DrawTestPic(tf.keras.callbacks.Callback):
    def __init__(self, J):
        self.J = J

    def on_batch_end(self, batch, logs=None):
        if batch % 800 == 0:
            self.J += 1
            V = VisualClass(path_to_h5)
            no = 72
            img_norm, GEO = V.get_norm_image(no, no + 1)
            predicted = model.predict(img_norm, verbose=False)
            predicted_classes = predicted.argmax(axis=-1)
            os.makedirs("./models/" + model_name + "/figures/fig" + str(no), exist_ok=True)
            f = V.draw_layers(no, predicted_classes)
            f.write_html("./models/" + model_name + "/figures/fig" + str(no) + "/" + str(self.J) + ".html")


class SaveClassificator(tf.keras.callbacks.Callback):
    def __init__(self, classificator_model):
        self.classificator = classificator_model

    def on_epoch_end(self, epoch, logs=None):
        w = model.get_layer("ResNet").get_weights()
        self.classificator.set_weights(w)
        self.classificator.save('./models/' + model_name + '/classificator_last')


callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, min_delta=3e-3),
    tf.keras.callbacks.TensorBoard(log_dir=logdir),
    DrawTestPic(J=0),
    SaveClassificator(classificator_model=classificator)
]

model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=LR))

train_dataset = make_train_dataset(path_to_h5, BATCH_SIZE, WIDTH, HEIGHT, CHANNELS)
history = model.fit(train_dataset, epochs=15,
                    steps_per_epoch=NUM_OF_PICTURES // BATCH_SIZE //CUTTING,
                    callbacks=callbacks,
                    verbose=1)
model.save('./models/' + model_name + '/last')
print('Model' + model_name + 'has been trained.')
