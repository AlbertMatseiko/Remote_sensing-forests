import tensorflow as tf

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


def encoder_block(x, num_filters, k_conv=3, s=2):
    x = conv_block(x, num_filters, k_conv)
    p = Conv2D(num_filters, k_conv, strides=s, padding="same")(x)
    p = BatchNormalization(axis=-1)(p)
    p = LeakyReLU(alpha=0.1)(x)
    return x, p


def decoder_block(x, skip_features, num_filters, k_conv=3, stride=2):
    x = Conv2DTranspose(num_filters, k_conv, strides=stride, padding="same")(x)
    x = Concatenate()([x, skip_features])
    x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = conv_block(x, num_filters, k_conv)
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


# Задаём фильтры и размеры ядер на этапе создания модели
def build_unet(filters, conv_kernel, strides, CHANNELS=7, CLASSES=10):
    inputs = Input(shape=(None, None, CHANNELS))

    s1, p1 = encoder_block(inputs, filters[0], conv_kernel[0], strides[0])
    s2, p2 = encoder_block(p1, filters[1], conv_kernel[1], strides[1])
    s3, p3 = encoder_block(p2, filters[2], conv_kernel[2], strides[2])
    s4, p4 = encoder_block(p3, filters[3], conv_kernel[3], strides[3])

    b1 = conv_block(p4, filters[4], conv_kernel[4])

    d1 = decoder_block(b1, s4, filters[3], conv_kernel[3], stride=strides[3])
    d2 = decoder_block(d1, s3, filters[2], conv_kernel[2], stride=strides[2])
    d3 = decoder_block(d2, s2, filters[1], conv_kernel[1], stride=strides[1])
    d4 = decoder_block(d3, s1, filters[0], conv_kernel[0], stride=strides[0])

    outputs = Conv2D(CLASSES, 1, padding="same", activation="softmax")(d4)

    model = Model(inputs, outputs, name="U-Net")
    return model


def build_resnet(filters, conv_kernel, CHANNELS=7, CLASSES=10):
    inputs = Input(shape=(None, None, CHANNELS))

    x = res_block(inputs, filters[0], conv_kernel[0])
    x = res_block(x, filters[1], conv_kernel[1])

    outputs = Conv2D(CLASSES, 1, padding="same", activation="softmax")(x)

    model = Model(inputs, outputs, name="ResNet")
    return model


def simple_classifier(CHANNELS=7, CLASSES=10):
    inp = tf.keras.layers.Input(shape=(None, None, CHANNELS))  # 512x512x7
    x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(inp)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    c = tf.keras.layers.Conv2D(CLASSES, 1, padding='same', activation='softmax')(x)
    return tf.keras.Model(inputs=inp, outputs=c)
