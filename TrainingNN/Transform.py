import tensorflow as tf
import numpy as np
WIDTH = 512
HEIGHT = 512

def generate_affine_transform_params(batch_size, size = WIDTH):
    pi = np.pi

    # generate rotation angle
    refl_index = tf.math.round(tf.random.uniform((batch_size,), minval=0, maxval=1, dtype=tf.float32))*2.-1.
    theta = tf.random.uniform((batch_size,), minval = -pi, maxval = pi, dtype = 'float32')
    
    c1 = tf.cos(theta)
    s1 = -tf.sin(theta)*refl_index
    s2 = tf.sin(theta)
    c2 = tf.cos(theta)*refl_index
    
    rot_matrix = tf.convert_to_tensor([[c1, s1],[s2, c2] ])
    rot_matrix = tf.transpose( rot_matrix, perm=(2,0,1) )
    inv_rot_matrix = tf.convert_to_tensor([ [c1, s2],[s1, c2] ])
    inv_rot_matrix = tf.transpose( inv_rot_matrix, perm=(2,0,1) )
    c_fin = tf.convert_to_tensor([ [size/np.sqrt(2)], [size/np.sqrt(2)] ], dtype = 'float32')
    c_in = tf.convert_to_tensor([[ size/2], [size/2] ], dtype = 'float32')
    c_rot = tf.linalg.matmul( rot_matrix, c_in  )
    delta = tf.linalg.matmul( inv_rot_matrix, c_fin - c_rot )
    x_shift = -delta[:,0,0]
    y_shift = -delta[:,1,0]
    
    # For (x, y) = T(x', y'):
    inverse_params = tf.stack([
        c1,  s1, -x_shift*c1-y_shift*s1,
        s2,  c2, -y_shift*c2-x_shift*s2,
        tf.zeros(batch_size), tf.zeros(batch_size)
    ], axis = 1)

    # For (x', y') = T(x, y):
    params = tf.stack([
         c1, s2, x_shift,
         s1, c2, y_shift,
        tf.zeros(batch_size), tf.zeros(batch_size)
    ], axis = 1)
    return params, inverse_params

class RandomAffineTransformParams(tf.keras.layers.Layer):
    def call(self, inp, WIDTH = 512):
        return generate_affine_transform_params(tf.shape(inp)[0], size = WIDTH)
    
class ImageProjectiveTransformLayer(tf.keras.layers.Layer):
    def __init__(self, interpolation='BILINEAR', fill_value=0, **kwargs):
        super(ImageProjectiveTransformLayer, self).__init__(**kwargs)
        self.interpolation = interpolation
        self.fill_value = fill_value

    def call(self, inputs, transforms, WIDTH = 512, HEIGHT = 512):
        S1,S2 = WIDTH, HEIGHT
        if inputs.shape[1] != None and inputs.shape[1] > S1:
            return tf.raw_ops.ImageProjectiveTransformV3(
                images=inputs,
                transforms=transforms,
                output_shape=tuple((S1,S2)),
                interpolation=self.interpolation,
                fill_value=self.fill_value
            )
        else:
            S1 =int(S1*np.sqrt(2))+12
            S2 =int(S2*np.sqrt(2))+12
            out_shape = tuple((S1, S2))
            return tf.raw_ops.ImageProjectiveTransformV3(
                images=inputs,
                transforms=transforms,
                output_shape=out_shape,
                interpolation=self.interpolation,
                fill_value=self.fill_value
            )