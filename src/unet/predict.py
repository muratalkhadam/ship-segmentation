import numpy as np
from tensorflow import keras
from .metrics import dice_loss, dice_coef


def predict(model, test_ds):
    """
    Generates predictions using a trained model on the given test dataset.

    Args:
        model (str): The path to the saved model file.
        test_ds: The test dataset.

    Returns:
        np.ndarray: The prediction for each image.
    """
    threshold = 0.5

    model = keras.models.load_model(model, custom_objects={'dice_loss': dice_loss,
                                                           'dice_coef': dice_coef})
    predictions = model.predict(test_ds)
    predictions = np.where(predictions > threshold, 1, 0)
    return predictions
