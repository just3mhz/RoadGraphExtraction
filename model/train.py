import argparse

import tensorflow as tf

from unet import resnet50_unet
from dataset import make_dataset
from loss import dice_loss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', help='Folder with training dataset')
    parser.add_argument('--image-size', type=int, default=512, help='Images size')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--steps-per-epoch', type=int, default=80)
    parser.add_argument('--save-model', type=str, help='Path to save model')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    dataset = make_dataset(args.folder, (args.image_size, args.image_size), args.batch_size)
    model = resnet50_unet((args.image_size, args.image_size, 3))
    model.compile(
        optimizer='adam',
        loss=dice_loss,
        metrics=['accuracy'])
    model.fit(dataset, epochs=args.epochs, steps_per_epoch=args.steps_per_epoch)

    if args.save_model:
        tf.keras.models.save_model(model, args.save_model, save_format='tf')
