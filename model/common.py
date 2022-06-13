import tensorflow as tf
from tensorflow.keras import backend

SMOOTH = 1e-6


def unstack(tensor, axis=3, image_size=512):
    unstacked_tensors = []
    for i, t in enumerate(tf.unstack(tensor, axis=axis)):
        unstacked_tensors.append(tf.reshape(t, [-1, image_size, image_size, 1]))
    return unstacked_tensors


def _gather_channels(x, indexes):
    if backend.image_data_format() == 'channels_last':
        x = backend.permute_dimensions(x, (3, 0, 1, 2))
        x = backend.gather(x, indexes)
        x = backend.permute_dimensions(x, (1, 0, 2, 3))
    else:
        x = backend.permute_dimensions(x, (1, 0, 2, 3))
        x = backend.gather(x, indexes)
        x = backend.permute_dimensions(x, (1, 0, 2, 3))
    return x


def gather_channels(*xs, indexes=None):
    if indexes is None:
        return xs
    elif isinstance(indexes, int):
        indexes = [indexes]
    xs = [_gather_channels(x, indexes=indexes) for x in xs]
    return xs


def round_if_needed(x, threshold):
    if threshold is not None:
        x = backend.greater(x, threshold)
        x = backend.cast(x, backend.floatx)
    return x


def get_reduce_axes(per_image):
    axes = [1, 2] if backend.image_data_format() == 'channels_last' else [2, 3]
    if not per_image:
        axes.insert(0, 0)
    return axes


def average(x, per_image=False, class_weights=None):
    if per_image:
        x = backend.mean(x, axis=0)
    if class_weights is not None:
        x = x * class_weights
    return backend.mean(x)
