import numpy as np
from tensorflow import keras
from .metrics import dice_loss, dice_coef


def inference(model, x_test, y_test):
    model = keras.models.load_model(model, custom_objects={'dice_loss': dice_loss,
                                                           'dice_coef': dice_coef})

    loss, score = model.evaluate(x_test, y_test)
    return loss, score
