import os
import cv2
import numpy as np


def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])


def labelcolormap(N):
    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = 0
        g = 0
        b = 0
        id = i
        for j in range(7):
            str_id = uint82bin(id)
            r = r ^ (np.uint8(str_id[-1]) << (7 - j))
            g = g ^ (np.uint8(str_id[-2]) << (7 - j))
            b = b ^ (np.uint8(str_id[-3]) << (7 - j))
            id = id >> 3
        cmap[i, 0] = b
        cmap[i, 1] = g
        cmap[i, 2] = r
    return cmap


colors = labelcolormap(256)


def show_frame(pred, image=None, out_file='', vis=False):
    if vis:
        result = np.dstack((colors[pred, 0], colors[pred, 1],
                            colors[pred, 2])).astype(np.uint8)

    if out_file != '':
        if not os.path.exists(os.path.split(out_file)[0]):
            os.makedirs(os.path.split(out_file)[0])
        if vis:
            cv2.imwrite(out_file, result)
        else:
            cv2.imwrite(out_file, pred)

    if vis and image is not None:
        temp = image.astype(float) * 0.4 + result.astype(float) * 0.6
        cv2.imshow('Result', temp.astype(np.uint8))
        cv2.waitKey()
