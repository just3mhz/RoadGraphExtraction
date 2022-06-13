import pickle
import logging

from typing import Tuple
from typing import List
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

from gte import GraphTensorEncoder


def to_int(u):
    return min(int(u[1]), 1299), min(int(u[0]), 1299)


def load_graph(path):
    graph = defaultdict(set)
    for u, neighbours in pickle.load(open(path, 'rb')).items():
        u = to_int(u)
        for v in neighbours:
            graph[u].add(to_int(v))
    return graph


def rescale_point(point, old_image_size, new_image_size):
    x = min(int(point[0] * new_image_size / old_image_size), new_image_size - 1)
    y = min(int(point[1] * new_image_size / old_image_size), new_image_size - 1)
    return x, y


def rescale_graph(graph, old_image_size, new_image_size):
    rescaled_graph = defaultdict(set)
    for u, neighbours in graph.items():
        u = rescale_point(u, old_image_size, new_image_size)
        for v in neighbours:
            v = rescale_point(v, old_image_size, new_image_size)
            rescaled_graph[u].add(v)
    return rescaled_graph


class GteDatasetReader:
    def __init__(self, df: pd.DataFrame, image_size: Tuple[int, int]):
        self.df = df
        self.image_size = image_size
        self.old_image_size = 0
        self.encoder = GraphTensorEncoder(image_size=image_size[0],
                                          max_degree=6, d=25,
                                          pv_threshold=0.3,
                                          pe_threshold=0.3)

    def __call__(self):
        for index, row in self.df.iterrows():
            tile = self._read_tile(row['tile'])
            graph = self._read_graph(row['graph'])
            yield tile, graph

    def _read_tile(self, path):
        tile = cv2.imread(path)
        self.old_image_size = tile.shape[0:2]
        if tile.shape[:2] != self.image_size:
            tile = cv2.resize(tile, self.image_size)
        return tile

    def _read_graph(self, path):
        graph = load_graph(path)
        graph = rescale_graph(graph, self.old_image_size[0], self.image_size[0])
        return self.encoder.encode(graph)


class SegmentationDatasetReader:
    def __init__(self, df: pd.DataFrame, image_size: Tuple[int, int]):
        self.df = df
        self.image_size = image_size

    def __call__(self):
        for index, row in self.df.iterrows():
            tile = self._read_tile(row['tile'])
            mask = self._read_mask(row['mask'])
            yield tile, mask

    def _read_mask(self, path):
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask.shape[:2] != self.image_size:
            mask = cv2.resize(mask, self.image_size)
        mask = np.where(mask > 0, 1.0, 0.0)
        mask = mask.reshape(self.image_size + (1,))
        return mask

    def _read_tile(self, path):
        tile = cv2.imread(path)
        if tile.shape[:2] != self.image_size:
            tile = cv2.resize(tile, self.image_size)
        return tile


class RandomFlip(layers.Layer):
    def __init__(self, seed=42):
        super().__init__()
        self.augment_tile = layers.RandomFlip(seed=seed)
        self.augment_mask = layers.RandomFlip(seed=seed)


    def call(self, tile, mask):
        tile = self.augment_tile(tile)
        mask = self.augment_mask(mask)
        return tile, mask


def make_gte_dataset(datasets: List[str],
                 image_size: Tuple[int, int],
                 batch_size: int,
                 random_seed: int = 42,
                 val_split: float = 0.05):

    df = pd.concat([pd.read_csv(dataset, index_col=0) for dataset in datasets])
    train, test = train_test_split(df, test_size=val_split)

    output_signature = (tf.TensorSpec(shape=image_size + (3,), dtype=tf.float32),
                        tf.TensorSpec(shape=image_size + (19,), dtype=tf.float32))

    reader = GteDatasetReader(train, image_size)
    dataset = tf.data.Dataset.from_generator(reader, output_signature=output_signature)
    dataset = dataset.cache()
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()

    reader = GteDatasetReader(test, image_size)
    validation = tf.data.Dataset.from_generator(reader, output_signature=output_signature)
    validation = validation.cache()
    validation = validation.batch(8)

    return dataset, validation


def make_segmentation_dataset(datasets: List[str],
                 image_size: Tuple[int, int],
                 batch_size: int,
                 random_seed: int = 42,
                 val_split: float = 0.05):

    df = pd.concat([pd.read_csv(dataset, index_col=0) for dataset in datasets])
    train, test = train_test_split(df, test_size=val_split)

    output_signature = (tf.TensorSpec(shape=image_size + (3,), dtype=tf.float32),
                        tf.TensorSpec(shape=image_size + (1,), dtype=tf.float32))

    reader = SegmentationDatasetReader(train, image_size)
    dataset = tf.data.Dataset.from_generator(reader, output_signature=output_signature)
    dataset = dataset.cache()
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()

    reader = SegmentationDatasetReader(test, image_size)
    validation = tf.data.Dataset.from_generator(reader, output_signature=output_signature)
    validation = validation.cache()
    validation = validation.batch(8)

    return dataset, validation
