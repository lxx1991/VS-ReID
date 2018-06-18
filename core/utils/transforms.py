import random
import cv2
import numpy as np


def crop(img, offeset):
    h1, w1, h2, w2 = offeset
    return img[h1:h2, w1:w2, ...]


def random_crop(img, target_size):
    h, w = img.shape[0:2]
    th, tw = target_size
    h1 = random.randint(0, max(0, h - th))
    w1 = random.randint(0, max(0, w - tw))
    h2 = min(h1 + th, h)
    w2 = min(w1 + tw, w)
    return crop(img, [h1, w1, h2, w2])


def group_random_crop(img_group, target_size):
    h, w = img_group[0].shape[0:2]
    th, tw = target_size
    h1 = random.randint(0, max(0, h - th))
    w1 = random.randint(0, max(0, w - tw))
    h2 = min(h1 + th, h)
    w2 = min(w1 + tw, w)
    outs = list()
    for img in img_group:
        assert (img.shape[0] == h and img.shape[1] == w)
        outs.append(crop(img, [h1, w1, h2, w2]))
    return outs


def center_crop(img, target_size):
    h, w = img.shape[0:2]
    th, tw = target_size
    h1 = max(0, int((h - th) / 2))
    w1 = max(0, int((w - tw) / 2))
    h2 = min(h1 + th, h)
    w2 = min(w1 + tw, w)
    return crop(img, [h1, w1, h2, w2])


def group_center_crop(img_group, target_size):
    h, w = img_group[0].shape[0:2]
    th, tw = target_size
    h1 = max(0, int((h - th) / 2))
    w1 = max(0, int((w - tw) / 2))
    h2 = min(h1 + th, h)
    w2 = min(w1 + tw, w)
    outs = list()
    for img in img_group:
        assert (img.shape[0] == h and img.shape[1] == w)
        outs.append(crop(img, [h1, w1, h2, w2]))
    return outs


def pad(img, offeset, value=0):
    h1, w1, h2, w2 = offeset
    img = cv2.copyMakeBorder(
        img, h1, h2, w1, w2, cv2.BORDER_CONSTANT, value=value)
    return img


def random_pad(img, target_size, value=0):
    h, w = img.shape[0:2]
    th, tw = target_size

    h1 = random.randint(0, max(0, th - h))
    w1 = random.randint(0, max(0, tw - w))
    h2 = max(th - h - h1, 0)
    w2 = max(tw - w - w1, 0)
    return pad(img, [h1, w1, h2, w2], value=value)


def group_random_pad(img_group, target_size, values):
    h, w = img_group[0].shape[0:2]
    th, tw = target_size

    h1 = random.randint(0, max(0, th - h))
    w1 = random.randint(0, max(0, tw - w))
    h2 = max(th - h - h1, 0)
    w2 = max(tw - w - w1, 0)
    outs = list()
    for img, value in zip(img_group, values):
        assert (img.shape[0] == h and img.shape[1] == w)
        outs.append(pad(img, [h1, w1, h2, w2], value=value))
    return outs


def center_pad(img, target_size, value=0):
    h, w = img.shape[0:2]
    th, tw = target_size

    h1 = max(0, int((th - h) / 2))
    w1 = max(0, int((tw - w) / 2))
    h2 = max(th - h - h1, 0)
    w2 = max(tw - w - w1, 0)
    return pad(img, [h1, w1, h2, w2], value=value)


def group_center_pad(img_group, target_size, values):
    h, w = img_group[0].shape[0:2]
    th, tw = target_size

    h1 = max(0, int((th - h) / 2))
    w1 = max(0, int((tw - w) / 2))
    h2 = max(th - h - h1, 0)
    w2 = max(tw - w - w1, 0)
    outs = list()

    for img, value in zip(img_group, values):
        assert (img.shape[0] == h and img.shape[1] == w)
        outs.append(pad(img, [h1, w1, h2, w2], value=value))
    return outs


def group_concer_pad(img_group, target_size, values):
    h, w = img_group[0].shape[0:2]
    th, tw = target_size

    h1 = 0
    w1 = 0
    h2 = max(th - h - h1, 0)
    w2 = max(tw - w - w1, 0)
    outs = list()

    for img, value in zip(img_group, values):
        assert (img.shape[0] == h and img.shape[1] == w)
        outs.append(pad(img, [h1, w1, h2, w2], value=value))
    return outs


def rescale(img, scales, interpolation=cv2.INTER_LINEAR):
    if isinstance(scales, list):
        if len(scales) == 2:
            scale = random.uniform(scales[0], scales[1])
        else:
            scale = random.choice(scales)
    else:
        scale = scales

    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=interpolation)

    return img


def group_rescale(img_group, scales, interpolations):
    if isinstance(scales, list):
        if len(scales) == 2:
            scale = random.uniform(scales[0], scales[1])
        else:
            scale = random.choice(scales)
    else:
        scale = scales

    outs = list()
    for img, interpolation in zip(img_group, interpolations):
        outs.append(rescale(img, scale, interpolation))

    return outs


def rotation(img, degrees, interpolation=cv2.INTER_LINEAR, value=0):
    if isinstance(degrees, list):
        if len(degrees) == 2:
            degree = random.uniform(degrees[0], degrees[1])
        else:
            degree = random.choice(degrees)
    else:
        degree = degrees

    h, w = img.shape[0:2]
    center = (w / 2, h / 2)
    map_matrix = cv2.getRotationMatrix2D(center, degree, 1.0)

    img = cv2.warpAffine(
        img,
        map_matrix, (w, h),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=value)

    return img


def group_rotation(img_group, degrees, interpolations, values):
    if isinstance(degrees, list):
        if len(degrees) == 2:
            degree = random.uniform(degrees[0], degrees[1])
        else:
            degree = random.choice(degrees)
    else:
        degree = degrees

    outs = list()
    for img, interpolation, value in zip(img_group, interpolations, values):
        outs.append(rotation(img, degree, interpolation, value))
    return outs


def flip(img):
    return np.fliplr(img)


def random_flip(img):
    if random.random() < 0.5:
        return flip(img)
    else:
        return img


def group_flip(img_group):
    return [flip(img) for img in img_group]


def group_random_flip(img_group):
    if random.random() < 0.5:
        return [flip(img) for img in img_group]
    else:
        return img_group


def normalize(img, mean, std=None):
    img = img - np.array(mean)[np.newaxis, np.newaxis, ...]
    if std is not None:
        img = img / np.array(std)[np.newaxis, np.newaxis, ...]
    return img


def blur(img, kenrel_size=(5, 5), sigma=(1e-6, 0.6)):
    img = cv2.GaussianBlur(img, kenrel_size, random.uniform(*sigma))
    return img


def random_blur(img, kenrel_size=(5, 5), sigma=(1e-6, 0.6)):
    if random.random() < 0.5:
        return blur(img, kenrel_size, sigma)
    else:
        return img
