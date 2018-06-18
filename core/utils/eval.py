import numpy as np


class EvalSegmentation(object):
    def __init__(self, num_class, ignore_label=None):
        self.num_class = num_class
        self.ignore_label = ignore_label
        self.clear()

    def __call__(self, pred, gt):
        assert (pred.shape == gt.shape)
        gt = gt.flatten().astype(int)
        pred = pred.flatten().astype(int)
        locs = np.bitwise_and((gt != self.ignore_label),
                              (pred != self.ignore_label))
        sumim = gt + pred * self.num_class
        hs = np.bincount(
            sumim[locs], minlength=self.num_class**2).reshape(
                self.num_class, self.num_class)
        self.conf += hs

    def acc(self):
        return np.sum(np.diag(self.conf)) / float(np.sum(self.conf))

    def num(self):
        return np.sum(self.conf, axis=1)

    def IoU(self):
        return np.diag(self.conf) / (
            1e-20 + self.conf.sum(1) + self.conf.sum(0) - np.diag(self.conf))

    def mIoU(self):
        iou = self.IoU()
        return np.sum(iou) / len(iou)

    def clear(self):
        self.conf = np.zeros((self.num_class, self.num_class))
