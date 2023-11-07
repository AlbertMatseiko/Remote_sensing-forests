import tensorflow as tf

if __name__ != "__main__":
    import sys

    sys.path.append("../")

from TrainingNN.Transform import ImageProjectiveTransformLayer, RandomAffineTransformParams
from TrainingNN.Loss import conv_loss, negative_mutual_inf_with_shifts, negative_mutual_inf_without_shifts

# Strange import because of PyCharm's bug
tfl = tf.keras.layers
Conv2D = tfl.Conv2D
BatchNormalization = tfl.BatchNormalization
Activation = tfl.Activation
LeakyReLU = tfl.LeakyReLU
MaxPool2D = tfl.MaxPool2D
Conv2DTranspose = tfl.Conv2DTranspose
Concatenate = tfl.Concatenate
Input = tfl.Input
Add = tfl.Add
Model = tf.keras.models.Model


def conv_block(x, num_filters, kernel):
    x = Conv2D(num_filters, kernel, padding="same")(x)
    x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x


def res_block(x, f, k):
    x_skip = x
    x_skip = conv_block(x_skip, f, k)
    x = conv_block(x, f, k)
    x = conv_block(x, f, k)
    x = Add()([x, x_skip])
    x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x


def build_resnet(filters=None, conv_kernel=None, depth=2, CHANNELS=7, CLASSES=10):
    if conv_kernel is None:
        conv_kernel = [3, 3, 3]
    if filters is None:
        filters = [32, 64, 32]
    inputs = Input(shape=(None, None, CHANNELS))
    x = inputs
    for i in range(depth):
        x = res_block(x, filters[i], conv_kernel[i])

    outputs = Conv2D(CLASSES, 1, padding="same", activation="softmax")(x)

    model = Model(inputs, outputs, name="ResNet")
    return model


### Задаём фильтры и размеры ядер на этапе создания модели
### Список 'filters' - кол-во фильтров, по порядку следования слоёв 'encoder'
### Список 'conv_kernels' - размер ядер свёрток в 'encoder' и 'decoder', по порядку следования слоёв 'encoder'
### Список 'strides' - размер 'strides' в 'encoder' и 'decoder', по порядку следования слоёв 'encoder'
def make_resnet_model(filters: list = None, conv_kernel: list = None, depth=2,
                      apply_conv_loss=False, apply_shifts=False,
                      CHANNELS=7, CLASSES=10, WIDTH=256, HEIGHT=256, BATCH_SIZE=8,
                      CONTRAST_FACTOR=0.1):
    # Создаём основу модели
    if conv_kernel is None:
        conv_kernel = [3, 3, 3]
    if filters is None:
        filters = [32, 64, 32]
    inp = tf.keras.layers.Input(shape=(None, None, CHANNELS))

    # classifier = simple_classifier()
    classifier = build_resnet(filters, conv_kernel, depth, CHANNELS, CLASSES)
    # classifier = build_unet(filters, conv_kernel, strides)

    outp = classifier(inp)
    model = tf.keras.Model(inputs=inp, outputs=outp)

    # По гиперпараметрам генерируем имя модели
    s = 'f'
    for i in filters:
        s += '.' + str(i)
    s += '_k'
    for i in conv_kernel:
        s += '.' + str(i)
    # s += '_s'
    # for i in strides:
    #    s +='.'+str(i)
    s += '_c' + str(CONTRAST_FACTOR)

    if apply_conv_loss:
        loss_fun = conv_loss
        loss_name = 'conv'
    elif apply_shifts:
        loss_fun = negative_mutual_inf_with_shifts
        loss_name = 'nmi_with_shifts'
    else:
        loss_fun = negative_mutual_inf_without_shifts
        loss_name = 'nmi_without_shifts'
    model_name = (str(classifier.name) + '_' + s + '_CLASSES.' + str(CLASSES) + '_BS.' + str(BATCH_SIZE)
                  + '_loss.' + loss_name)

    # Алгоритм подсчёта лосса
    params, inverse_params = RandomAffineTransformParams()(inp, WIDTH)
    transformed_inp = ImageProjectiveTransformLayer()(inp, params, WIDTH, HEIGHT)
    transformed_inp = tf.image.random_contrast(transformed_inp, 1. - CONTRAST_FACTOR, 1. + CONTRAST_FACTOR)
    transformed_outp = classifier(transformed_inp)
    inv_transformed_outp = ImageProjectiveTransformLayer()(transformed_outp, inverse_params)
    model.add_loss(loss_fun(outp, inv_transformed_outp, WIDTH=WIDTH, HEIGHT=HEIGHT,
                            BATCH_SIZE=BATCH_SIZE))
    return model, model_name
