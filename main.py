import h5py
from datetime import datetime
import os

#importing tensorflow, check gpu
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'), tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    print(gpu)
    tf.config.experimental.set_memory_growth(gpu, True)

#importing from local scripts
from TrainingNN.DataLoad import BatchLoader, make_train_dataset
from TrainingNN.BuildNN import build_resnet
from TrainingNN.Transform import *
from TrainingNN.Loss import conv_loss
from TrainingNN.Visualize import VisualClass


path_to_h5 = '../DATA/data/LC08_L2SP_02_T1_cropped_small.h5'
with h5py.File(path_to_h5, 'r') as f:
    LENGTH_OF_EPOCH = len(f['all/data_norm'])//5
    print('Number of small images in h5:', LENGTH_OF_EPOCH)

#Global variables, do not change
WIDTH = 256
HEIGHT = 256
CHANNELS = 7
CLASSES = 10
MAX_SHIFT = 1 # максимальное смещение по вертикали и горизонтали в функции потерь
BATCH_SIZE = 16

###Задаём фильтры и размеры ядер на этапе создания модели
###Список 'filters' - кол-во фильтров, по порядку следования слоёв 'encoder'
###Список 'conv_kernels' - размер ядер свёрток в 'encoder' и 'decoder', по порядку следования слоёв 'encoder'
###Список 'strides' - размер 'strides' в 'encoder' и 'decoder', по порядку следования слоёв 'encoder'
def make_model(filters = [32,32,32], conv_kernel = [3,3,3]):#, strides = [2,2,2,2]):
    
    #Создаём основу модели
    inp = tf.keras.layers.Input(shape=(None, None, CHANNELS))
    
    #classifier = simple_classifier()
    classifier = build_resnet(filters, conv_kernel, CHANNELS, CLASSES)
    #classifier = build_unet(filters, conv_kernel, strides)

    outp = classifier(inp)
    model = tf.keras.Model(inputs=inp, outputs=outp)
    
    #По гиперпараметрам генерируем имя модели
    s = 'f'
    for i in filters:
        s +='.'+str(i)
    s+='_k'
    for i in conv_kernel:
        s +='.'+str(i)
    s+='_s'
    #for i in strides:
    #    s +='.'+str(i)
    
    model_name = str(classifier.name)+'_'+s+'_CLASSES.'+str(CLASSES)+'_BS.'+str(BATCH_SIZE)
    
    #Алгоритм подсчёта лосса
    params, inverse_params = RandomAffineTransformParams()(inp, WIDTH)
    transformed_inp = ImageProjectiveTransformLayer()(inp, params, WIDTH, HEIGHT)
    transformed_outp = classifier(transformed_inp)
    inv_transformed_outp = ImageProjectiveTransformLayer()(transformed_outp, inverse_params)
    model.add_loss(conv_loss(outp, inv_transformed_outp, WIDTH, HEIGHT, BATCH_SIZE))
    return model, model_name

model, model_name = make_model()
print(model_name)

# making dir for model if necessary
try:
    os.makedirs('./models/'+model_name)
    print('directory for the model is created')
except:
    print('directory for the model already exists')
#make a dir for tensorboard logs
logdir = "./models/logs_tb/"+model_name+"/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(logdir)
print('directory for tb logs is created')
    

#Make callbacks: draw a pic after every epoch, early stopping, model checkpoint, logs to tensorboard
class DrawTestPic(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        V = VisualClass(path_to_h5)
        img_norm, GEO = V.get_norm_image(0,10)
        predicted = model.predict(img_norm, verbose = False)
        predicted_classes = predicted.argmax(axis = -1)
        no = 6
        try:
            os.makedirs("./models/"+model_name+"/figures/fig"+str(no))
        except:
            pass
        f = V.draw_layers(no, predicted_classes)
        f.write_html("./models/"+model_name+"/figures/fig"+str(no)+"/"+str(epoch)+".html")

callbacks = [
                tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, min_delta=5e-4),
                tf.keras.callbacks.ModelCheckpoint(filepath='../models/' + model_name + '/best',
                                                   monitor = 'loss',
                                                   save_freq='epoch'), 
                tf.keras.callbacks.TensorBoard(log_dir=logdir),
                DrawTestPic()
            ]

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))

train_dataset = make_train_dataset(path_to_h5, BATCH_SIZE, WIDTH, HEIGHT, CHANNELS)
history = model.fit(train_dataset, epochs = 15,
                    steps_per_epoch = LENGTH_OF_EPOCH // BATCH_SIZE,
                    callbacks=callbacks,
                    verbose = 1)
model.save('./models/'+ model_name + '/last')
print('Model' + model_name + 'has been trained.')