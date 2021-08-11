from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Add, Conv2DTranspose, BatchNormalization, Activation


def model():
    vgg = VGG16(include_top=False, input_shape=(416, 416, 3))

    x = vgg.output

    x = Conv2DTranspose(1024, (3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(1024, (3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(91, (3, 3), activation='sigmoid')(x)
    return Model(inputs=vgg.input, outputs=x)


def model_designed():
    inputs = Input(shape=(416, 416, 3))

    filters = 32
    skip_in = Conv2D(filters, (1, 1), padding='same', activation='relu')(inputs)
    x = Conv2D(filters, (3, 3), padding='same', activation='relu')(skip_in)
    skip_out = Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
    x = Add()([skip_in, skip_out])
    x = MaxPooling2D(pool_size=(2, 2))(x)

    filters = 64
    skip_in = Conv2D(filters, (1, 1), padding='same', activation='relu')(x)
    x = Conv2D(filters, (3, 3), padding='same', activation='relu')(skip_in)
    skip_out = Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
    x = Add()([skip_in, skip_out])
    x = MaxPooling2D(pool_size=(2, 2))(x)

    filters = 128
    skip_in = Conv2D(filters, (1, 1), padding='same', activation='relu')(x)
    x = Conv2D(filters, (3, 3), padding='same', activation='relu')(skip_in)
    skip_out = Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
    x = Add()([skip_in, skip_out])
    x = MaxPooling2D(pool_size=(2, 2))(x)

    filters = 256
    skip_in = Conv2D(filters, (1, 1), padding='same', activation='relu')(x)
    x = Conv2D(filters, (3, 3), padding='same', activation='relu')(skip_in)
    skip_out = Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
    x = Add()([skip_in, skip_out])
    x = MaxPooling2D(pool_size=(2, 2))(x)

    filters = 512
    skip_in = Conv2D(filters, (1, 1), padding='same', activation='relu')(x)
    x = Conv2D(filters, (3, 3), padding='same', activation='relu')(skip_in)
    skip_out = Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
    x = Add()([skip_in, skip_out])
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    outputs = Conv2D(91, (3, 3), activation='sigmoid')(x)
    return Model(inputs=inputs, outputs=outputs)


if __name__ == '__main__':
    model = model()
    model.summary()
