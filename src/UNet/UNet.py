from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout
from keras.models import Model


IMG_WIDTH = 768
IMG_HEIGHT = 768
IMG_CHANNELS = 3
DROPOUT_RATE = 0.5
POOL_SIZE = (2, 2)
KERNEL_SIZE = (3, 3)
STRIDES = POOL_SIZE


# UNet model
def create_unet(filters=64):
    inputs = Input((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))

    # Contraction path
    c1 = Conv2D(filters, KERNEL_SIZE, activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Dropout(DROPOUT_RATE)(c1)
    c1 = Conv2D(filters, KERNEL_SIZE, activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D(POOL_SIZE)(c1)

    c2 = Conv2D(filters * 2, KERNEL_SIZE, activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(DROPOUT_RATE)(c2)
    c2 = Conv2D(filters * 2, KERNEL_SIZE, activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D(POOL_SIZE)(c2)

    c3 = Conv2D(filters * 4, KERNEL_SIZE, activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(DROPOUT_RATE)(c3)
    c3 = Conv2D(filters * 4, KERNEL_SIZE, activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D(POOL_SIZE)(c3)

    c4 = Conv2D(filters * 8, KERNEL_SIZE, activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(DROPOUT_RATE)(c4)
    c4 = Conv2D(filters * 8, KERNEL_SIZE, activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(POOL_SIZE)(c4)

    # Bridge (1024)
    c5 = Conv2D(filters * 16, KERNEL_SIZE, activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(DROPOUT_RATE)(c5)
    c5 = Conv2D(filters * 16, KERNEL_SIZE, activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    # Expansive path
    u6 = Conv2DTranspose(filters * 8, POOL_SIZE, strides=STRIDES, padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(filters * 8, KERNEL_SIZE, activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(DROPOUT_RATE)(c6)
    c6 = Conv2D(filters * 8, KERNEL_SIZE, activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = Conv2DTranspose(filters * 4, POOL_SIZE, strides=STRIDES, padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(filters * 4, KERNEL_SIZE, activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(DROPOUT_RATE)(c7)
    c7 = Conv2D(filters * 4, KERNEL_SIZE, activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv2DTranspose(filters * 2, POOL_SIZE, strides=STRIDES, padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(filters * 2, KERNEL_SIZE, activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(DROPOUT_RATE)(c8)
    c8 = Conv2D(filters * 2, KERNEL_SIZE, activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = Conv2DTranspose(filters, POOL_SIZE, strides=STRIDES, padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(filters, KERNEL_SIZE, activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(DROPOUT_RATE)(c9)
    c9 = Conv2D(filters, KERNEL_SIZE, activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(c9)

    model = Model(inputs=[inputs], outputs=[outputs], name='unet')
    print(model.summary())

    return model
