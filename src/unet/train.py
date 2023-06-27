from .metrics import dice_loss, dice_coef
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from datetime import datetime


def train(model, X_train, Y_train, validation_size=0.2, epochs=30, batch_size=16, model_name='unet-segmentation'):
    """
        Trains a given model on the provided training data.

        Args:
            model: The model to be trained.
            X_train: The input training data.
            Y_train: The target training data.
            validation_size (float, optional): The fraction of training data to be used for validation, 0.2 by default.
            epochs (int, optional): The number of training epochs, 30 by default.
            batch_size (int, optional): The batch size for training, 16 by default.
            model_name (str, optional): The name of the model, 'unet-segmentation' by default.

        Returns:
            keras.callbacks.History: The training history.
        """
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=validation_size, random_state=42)

    model.compile(optimizer='adam', loss=[dice_loss], metrics=[dice_coef])
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    callbacks = [
        ModelCheckpoint(f'models/{model_name}{current_time}.h5', verbose=1, save_best_only=True),
        EarlyStopping(monitor='val_dice_coef', patience=5, mode='max')
    ]

    return model.fit(X_train, Y_train, validation_data=(X_val, Y_val),
                     epochs=epochs, batch_size=batch_size, callbacks=callbacks)
