import tensorflow as tf

from common import *


class FScoreByChannels:
    def __init__(self, channels=(0, ), image_size=512):
        self.f1_score = FScore()
        self.channels = channels
        self.image_size = image_size

    def __call__(self, gt, pr):
        gt = unstack(gt, axis=3, image_size=self.image_size)
        pr = unstack(pr, axis=3, image_size=self.image_size)
        value = 0
        for channel in self.channels:
            value += self.f1_score(gt[channel], pr[channel])
        return value / len(self.channels)

    def get_config(self):
        return {
            'channels': self.channels,
            'image_size': self.image_size
        }

    @classmethod
    def from_config(cls, config):
        print('???')
        return cls(**config)

class IouScore:
    def __init__(self, class_weights=1., class_indexes=None, smooth=SMOOTH, per_image=False, threshold=None):
        self.class_weights = class_weights
        self.class_indexes = class_indexes
        self.smooth = smooth
        self.per_image = per_image
        self.threshold = threshold

    def __call__(self, gt, pr):
        gt, pr = gather_channels(gt, pr, indexes=self.class_indexes)
        pr = round_if_needed(pr, self.threshold)
        axes = get_reduce_axes(self.per_image)

        intersection = backend.sum(gt * pr, axis=axes)
        union = backend.sum(gt + pr, axis=axes) - intersection

        score = (intersection + self.smooth) / (union + self.smooth)
        score = average(score, self.per_image, self.class_weights)

        return score


class FScore:
    def __init__(self, beta=1, class_weights=1, class_indexes=None, smooth=SMOOTH, per_image=False, threshold=None):
        self.beta = beta
        self.class_weights = class_weights
        self.class_indexes = class_indexes
        self.smooth = smooth
        self.per_image = per_image
        self.threshold = threshold

    def __call__(self, gt, pr):
        gt, pr = gather_channels(gt, pr, indexes=self.class_indexes)
        pr = round_if_needed(pr, self.threshold)
        axes = get_reduce_axes(self.per_image)

        # calculate score
        tp = backend.sum(gt * pr, axis=axes)
        fp = backend.sum(pr, axis=axes) - tp
        fn = backend.sum(gt, axis=axes) - tp

        score = ((1 + self.beta ** 2) * tp + self.smooth) / ((1 + self.beta ** 2) * tp + self.beta ** 2 * fn + fp + self.smooth)
        score = average(score, self.per_image, self.class_weights)

        return score


# Aliases
f1_score = FScore()
iou_score = IouScore()
