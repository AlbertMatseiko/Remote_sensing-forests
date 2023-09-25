import tensorflow as tf

def negative_mutual_inf_without_shifts(outp: tf.Tensor, inv_transformed_outp: tf.Tensor, 
                                       WIDTH = 512, HEIGHT = 512, BATCH_SIZE = 6):
    """Принимает на вход два тензора с размерностью BATCH * X * Y * CLASSES:
    - shift(pred(img)), где shift - произвольный сдвиг
    - shift_inv(T_inv(pred(T(img)))), где T - произвольная трансформация

    Собирая статистику по батчу и по пикселям, мы вычисляем отрицательную взаимную информацию.
    """

    joint_p = \
        tf.einsum('bxyi, bxyj -> ij', outp, inv_transformed_outp) / BATCH_SIZE / WIDTH / HEIGHT
        # CLASSES * CLASSES

    # transpose and sum to get symmetric matrix
    joint_p_T = tf.transpose(joint_p, perm=[1, 0])
    joint_p = (joint_p + joint_p_T) / 2

    P_i = tf.math.reduce_sum(joint_p, axis=-1, keepdims=True) # CLASSES * 1
    P_j = tf.math.reduce_sum(joint_p, axis=-2, keepdims=True) # 1 * CLASSES

    eps = tf.keras.backend.epsilon()

    per_pixel_mutual_inf = tf.math.reduce_sum(
        joint_p * tf.math.log((P_i * P_j + eps) / (joint_p + eps)), # +eps, чтобы избежать деления на 0
        axis=(-1, -2)
    )
    return tf.math.reduce_mean(per_pixel_mutual_inf)

def negative_mutual_inf_with_shifts(outp: tf.Tensor, inv_transformed_outp: tf.Tensor):
    """Принимает на вход два тензора с размерностью BATCH * WIDTH * HEIGHT * CLASSES:
    - pred(img)
    - T_inv(pred(T(img))), где T - произвольная трансформация

    Сдвигаем изображения на dx, dy и вычисляем отрицательную взаимную информацию.
    Изображения при этом обрезаются. Затем усредняем полученные числа по всем сдвигам.
    """

    def shift(tensor: tf.Tensor, dx: int, dy: int):
        if dx >= 0 and dy >= 0:
            return tensor[:, dx:, dy:, :]
        elif dx >= 0 and dy < 0:
            return tensor[:, dx:, :dy, :]
        elif dx < 0 and dy >= 0:
            return tensor[:, :dx, dy:, :]
        elif dx < 0 and dy < 0:
            return tensor[:, :dx, :dy, :]

    def shifted_mutual_infs(MAX_SHIFT = 1):
        for dx in range(-MAX_SHIFT, MAX_SHIFT + 1):
            for dy in range(-MAX_SHIFT, MAX_SHIFT + 1):
                yield negative_mutual_inf_without_shifts(
                    shift(outp, dx, dy),
                    shift(inv_transformed_outp, -dx, -dy)
                )

    return tf.math.reduce_mean(list(shifted_mutual_infs()))

def conv_loss(te1: tf.Tensor, te2: tf.Tensor, 
              WIDTH = 512, HEIGHT = 512, BATCH_SIZE = 6):
    #print(te1, te2)
    te1 = tf.transpose( te1, (3,1,2,0) )
    te2 = tf.transpose( te2, (1,2,0,3) )
    te1 = tf.pad( te1, [[0,0],[1,1],[1,1],[0,0]] )
    re_conv = tf.nn.conv2d( te1, te2, strides=1, padding='VALID')
    joint_p = tf.squeeze(re_conv[:,1,1,:])/BATCH_SIZE/HEIGHT/WIDTH
    
    # transpose and sum to get symmetric matrix
    joint_p_T = tf.transpose(joint_p, perm=[1, 0])
    joint_p = (joint_p + joint_p_T) / 2

    P_i = tf.math.reduce_sum(joint_p, axis=-1, keepdims=True) # CLASSES * 1
    P_j = tf.math.reduce_sum(joint_p, axis=-2, keepdims=True) # 1 * CLASSES

    eps = tf.keras.backend.epsilon()

    per_pixel_mutual_inf = tf.math.reduce_sum(
        joint_p * tf.math.log((P_i * P_j + eps) / (joint_p + eps)), # +eps, чтобы избежать деления на 0
        axis=(-1, -2)
    )
    return tf.math.reduce_mean(per_pixel_mutual_inf)