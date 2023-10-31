from keras.layers import Conv2D, BatchNormalization, Activation, Conv2DTranspose, Concatenate, Input, Add
from keras.models import Model

def conv_block(x, num_filters, kernel):
    x = Conv2D(num_filters, kernel, padding="same")(x)
    x = BatchNormalization(axis = -1)(x)
    x = Activation("relu")(x)
    return x

def encoder_block(x, num_filters, k_conv = 3, s = 2):
    x = conv_block(x, num_filters, k_conv)
    p = Conv2D(num_filters, k_conv, strides = s, padding="same")(x)
    p = BatchNormalization(axis = -1)(p)
    p = Activation("relu")(p)
    return x, p

def decoder_block(x, skip_features, num_filters, k_conv = 3, stride = 2):
    x = Conv2DTranspose(num_filters, k_conv, strides=stride, padding="same")(x)
    x = Concatenate()([x, skip_features])
    x = BatchNormalization(axis = -1)(x)
    x = Activation("relu")(x)
    x = conv_block(x, num_filters, k_conv)
    return x

def res_block(x, f, k):
    x_skip = x
    x_skip = conv_block(x_skip, f, k)
    x = conv_block(x, f, k)
    x = conv_block(x, f, k)
    x = Add()([x, x_skip])
    x = BatchNormalization(axis = -1)(x)
    x = Activation("relu")(x)
    return x

#Задаём фильтры и размеры ядер на этапе создания модели
def build_unet(filters, conv_kernel, strides, CHANNELS = 7, CLASSES = 10):
    inputs = Input(shape = (None, None, CHANNELS))

    s1, p1 = encoder_block(inputs, filters[0], conv_kernel[0], strides[0])
    s2, p2 = encoder_block(p1, filters[1], conv_kernel[1], strides[1])
    s3, p3 = encoder_block(p2, filters[2], conv_kernel[2], strides[2])
    s4, p4 = encoder_block(p3, filters[3], conv_kernel[3], strides[3])

    b1 = conv_block(p4, filters[4], conv_kernel[4])

    d1 = decoder_block(b1, s4, filters[3], conv_kernel[3], stride = strides[3])
    d2 = decoder_block(d1, s3, filters[2], conv_kernel[2], stride = strides[2])
    d3 = decoder_block(d2, s2, filters[1], conv_kernel[1], stride = strides[1])
    d4 = decoder_block(d3, s1, filters[0], conv_kernel[0], stride = strides[0])

    outputs = Conv2D(CLASSES, 1, padding="same", activation="softmax")(d4)

    model = Model(inputs, outputs, name="U-Net")
    return model

def build_resnet(filters, conv_kernel, CHANNELS = 7, CLASSES = 10):
    inputs = Input(shape = (None, None, CHANNELS))

    x = res_block(inputs, filters[0], conv_kernel[0])
    x = res_block(x, filters[1], conv_kernel[1])

    outputs = Conv2D(CLASSES, 1, padding="same", activation="softmax")(x)

    model = Model(inputs, outputs, name="ResNet")
    return model

def simple_classifier(CHANNELS = 7, CLASSES = 10):
    inp = Input(shape = (None, None, CHANNELS)) # 512x512x7
    x = Conv2D(32, 3, padding = 'same', activation = 'relu')(inp)
    x = BatchNormalization()(x)
    x = Conv2D(32, 3, padding = 'same', activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, 3, padding = 'same', activation = 'relu')(x)
    x = BatchNormalization()(x)
    c = Conv2D(CLASSES, 1, padding = 'same', activation = 'softmax')(x)
    return Model(inputs = inp, outputs = c)
