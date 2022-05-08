import glob
import os
import re

import cv2
import numpy as np
import tensorflow as tf


class ImageReader:
    def __init__(self, folder, image_size):
        self.folder = folder
        self.image_size = image_size

    def __call__(self):
        images_path = os.path.join(self.folder, '*_8bit.tif')
        for image_path in glob.glob(images_path):
            match = re.fullmatch(r'.*/([a-zA-Z0-9_]*)_8bit\.tif', image_path)
            if not match:
                continue
            mask_path = os.path.join(self.folder, f'{match.group(1)}_mask.tif')
            if not os.path.exists(mask_path):
                continue
            image = cv2.resize(cv2.imread(image_path), self.image_size)
            mask = cv2.resize(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE), self.image_size)
            yield image, np.where(mask == 150, 1.0, 0.0)


class HorizontalRandomFlip(tf.keras.layers.Layer):
    def __init__(self, seed=42):
        super().__init__()
        # both use the same seed, so they'll make the same random changes.
        self.augment_inputs = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
        self.augment_labels = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)

    def call(self, inputs, labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        return inputs, labels


def make_dataset(folder, image_size, batch_size):
    return tf.data.Dataset.from_generator(
        ImageReader(folder, image_size),
        output_signature=(
            tf.TensorSpec(shape=image_size + (3,), dtype=tf.float32),
            tf.TensorSpec(shape=image_size, dtype=tf.float32)
        )).cache()\
          .batch(batch_size)\
          .repeat()\
          .map(HorizontalRandomFlip())\
          .prefetch(buffer_size=tf.data.AUTOTUNE)
