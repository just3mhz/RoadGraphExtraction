import tensorflow as tf
from tensorflow.keras import backend

from metrics import FScore
from common import SMOOTH
from common import unstack


def gt_loss(y_true, y_pred, max_degree=6, image_size=512):
    y_true = unstack(y_true)
    y_pred = unstack(y_pred)

    batch_size = tf.shape(y_true[0])[0]

    soft_mask = tf.reshape(y_true[0], (batch_size, image_size, image_size))
    #  Crossentropy loss for vertex channel
    pv_loss = tf.reduce_mean(binary_crossentropy(y_true[0], y_pred[0]))
    pe_loss = 0
    for i in range(max_degree):
        pe_crossentropy = binary_crossentropy(y_true[1 + 3*i], y_pred[1 + 3*i])
        # Apply only to keypoints!
        pe_loss += tf.reduce_mean(tf.multiply(soft_mask, pe_crossentropy))
    direction_loss = 0
    for i in range(max_degree):
        v1 = tf.concat(y_true[2 + 3*i:4 + 3*i], axis=3)
        v2 = tf.concat(y_pred[2 + 3*i:4 + 3*i], axis=3)
        # Apply only to keypoints!
        direction_loss += tf.reduce_mean(tf.multiply(y_true[0], tf.square(v2 - v1)))
    return pv_loss + pe_loss + direction_loss


class DiceLoss:
    def __init__(self, beta=1, class_weights=None, class_indexes=None, smooth=SMOOTH, per_image=False):
        self.f_score = FScore(beta, class_weights, class_indexes, smooth, per_image)

    def __call__(self, gt, pr):
        return 1 - self.f_score(gt, pr)


def binary_crossentropy(gt, pr):
    return backend.mean(backend.binary_crossentropy(gt, pr))


# Aliases
dice_loss = DiceLoss()
