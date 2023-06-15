import tensorflow as tf


# dice coefficient for unet model
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_d = tf.reshape(y_true, [-1])
    y_pred_d = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_d * y_pred_d)
    dice = (2. * intersection + smooth) / (tf.reduce_sum(y_true_d) + tf.reduce_sum(y_pred_d) + smooth)
    return dice


# dice coefficient loss for unet model
def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)
