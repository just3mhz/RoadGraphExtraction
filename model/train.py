import argparse

import tensorflow as tf
import segmentation_models as sm

from tensorflow.keras.optimizers import schedules

from unet import resnet50_unet
from unet import GteOutput
from dataset import make_gte_dataset

import losses
import metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dataset', action='append', help='Path to training dataframe')
    parser.add_argument('--image-size', type=int, default=512, help='Images size')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--steps-per-epoch', type=int, default=881)
    parser.add_argument('--save-model', type=str, help='Path to save model')
    parser.add_argument('--load-model', type=str, help='Path to pretrained model')
    return parser.parse_args()


def model_from_scratch():
    model = resnet50_unet((512, 512, 3))
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    vertexness_fscore=metrics.FScoreByChannels()
    model.compile(
        optimizer=optimizer,
        loss=losses.gt_loss,
        metrics=vertexness_fscore)
    return model


def pretrained_model(path):
    model = tf.keras.models.load_model(path, {
        'gt_loss': losses.gt_loss,
        'vertexness_fscore': metrics.FScoreByChannels(),
    }, compile=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    vertexness_fscore=metrics.FScoreByChannels()
    model.compile(
        optimizer=optimizer,
        loss=losses.gt_loss,
        metrics=vertexness_fscore)
    return model


if __name__ == '__main__':
    sm.set_framework('tf.keras')

    args = parse_args()
    image_size = (args.image_size, args.image_size)
    dataset, validation = make_gte_dataset(args.train_dataset, image_size, args.batch_size)

    if args.load_model:
        model = pretrained_model(args.load_model)
    else:
        model = model_from_scratch()

    model.fit(dataset, epochs=args.epochs,
              steps_per_epoch=args.steps_per_epoch, validation_data=validation)

    if args.save_model:
        tf.keras.models.save_model(model, args.save_model, save_format='tf')
