import tensorflow as tf


# dice coefficient for unet model
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_d = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_d = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_d * y_pred_d)
    dice = (2. * intersection + smooth) / (tf.reduce_sum(y_true_d) + tf.reduce_sum(y_pred_d) + smooth)
    return dice


# dice coefficient loss for unet model
def dice_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)
