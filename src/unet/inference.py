from tensorflow import keras
from .metrics import dice_loss, dice_coef


def inference(model, x_test, y_test):
    """
    Performs inference using a trained model on the given test data.

    Args:
        model (str): The path to the saved model file.
        x_test (list): The input test data.
        y_test (list): The target test data.

    Returns:
        tuple: A tuple containing the dice loss and dice score of the model on the test data.
    """
    model = keras.models.load_model(model, custom_objects={'dice_loss': dice_loss,
                                                           'dice_coef': dice_coef})

    loss, score = model.evaluate(x_test, y_test)
    return loss, score
