from .metrics import dice_loss, dice_coef
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split


def train(model, X_train, Y_train, validation_size=0.2, epochs=30, batch_size=32):
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=validation_size, random_state=42)

    model.compile(optimizer='adam', loss=[dice_loss], metrics=[dice_coef])

    callbacks = [
        ModelCheckpoint('unet-segmentation.h5', verbose=1, save_best_only=True),
        EarlyStopping(monitor='dice_coef', patience=7, mode='max')
    ]

    return model.fit(X_train, Y_train, validation_data=(X_val, Y_val),
                     epochs=epochs, batch_size=batch_size, callbacks=callbacks)
