import tensorflow as tf
from tensorflow.keras.metrics import binary_crossentropy
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet import ResNet50

from common import unstack

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


class GteOutput(layers.Layer):
    def __init__(self, max_degree=6, *args, **kwargs):
        super(GteOutput, self).__init__(*args, **kwargs)
        self.max_degree = max_degree

    def call(self, inputs):
        outputs = unstack(inputs)
        outputs[0] = tf.math.sigmoid(outputs[0])
        for i in range(self.max_degree):
            outputs[1 + 3*i] = tf.math.sigmoid(outputs[1 + 3*i])
        return tf.concat(outputs, axis=3)

    def get_config(self):
        return {
            'max_degree': self.max_degree,
        }

    @classmethod
    def from_config(cls, config):
        print('???')
        return cls(**config)


def resnet50_unet(input_shape, max_degree=6):
    inputs = layers.Input(shape=input_shape, name='input')

    # Pre-trained resnet50
    resnet50 = ResNet50(include_top=False, weights='imagenet', input_tensor=inputs)

    # Skip connections
    s1 = resnet50.get_layer('input').output                   # x1
    s2 = resnet50.get_layer('conv1_relu').output              # x2
    s3 = resnet50.get_layer('conv2_block3_out').output        # x4
    s4 = resnet50.get_layer('conv3_block4_out').output        # x8

    # Bridge
    b1 = resnet50.get_layer('conv4_block6_out').output

    # Decoder blocks
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    raw_outputs = layers.Conv2D(
        1 + 3 * max_degree, kernel_size=(3, 3),
        padding='same', activation='linear', name='raw_output')(d4)

    outputs = GteOutput(max_degree=max_degree, name='gte_outputs')(raw_outputs)

    return Model(inputs, outputs, name='resnet50_unet')


if __name__ == '__main__':
    model = resnet50_unet((512, 512, 3))
    model.summary()
