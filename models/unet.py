from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet import ResNet50


def conv_block(inputs, filters):
    x = layers.Conv2D(filters, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    return x


def decoder_block(inputs, skip_features, filters):
    x = layers.Conv2DTranspose(filters, (2, 2), strides=2, padding='same')(inputs)
    x = layers.Concatenate()([x, skip_features])
    x = conv_block(x, filters)
    return x


def resnet50_unet(input_shape):
    inputs = layers.Input(shape=input_shape, name='input')

    # Pre-trained resnet50
    resnet50 = ResNet50(include_top=False, weights='imagenet', input_tensor=inputs)

    # Skip connections
    s1 = resnet50.get_layer('input').output
    s2 = resnet50.get_layer('conv1_relu').output
    s3 = resnet50.get_layer('conv2_block3_out').output
    s4 = resnet50.get_layer('conv3_block4_out').output

    # Bridge
    b1 = resnet50.get_layer('conv4_block6_out').output

    # Decoder blocks
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = layers.Conv2D(1, 1, padding='same', activation='sigmoid', name='output')(d4)

    return Model(inputs, outputs, name='resnet50_unet')


if __name__ == '__main__':
    model = resnet50_unet((512, 512, 3))
    model.summary()
