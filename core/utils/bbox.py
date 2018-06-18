import numpy as np


def gen_bbox(label, instance_list, enlarge=False, ratio=1.0):
    bbox = np.zeros((len(instance_list), 4), float)
    bbox_enlarge = 0.15 if enlarge else 0.0

    for i in instance_list:
        [x, y] = np.where(label[:, :] == i + 1)
        if len(y) > 0:
            y = sorted(y)
            x = sorted(x)
            wmin = y[int((len(y) - 1) * (1 - ratio))]
            wmax = y[int((len(y) - 1) * (ratio))] + 1
            hmin = x[int((len(x) - 1) * (1 - ratio))]
            hmax = x[int((len(x) - 1) * (ratio))] + 1
        else:
            bbox[i, :] = [0, 0, 1, 1]
            continue

        bbox_h = hmax - hmin
        bbox_w = wmax - wmin

        wmin = np.clip((wmin - bbox_enlarge * bbox_w), 0, label.shape[1] - 1)
        wmax = np.clip((wmax + bbox_enlarge * bbox_w), wmin + 1, label.shape[1])
        hmin = np.clip((hmin - bbox_enlarge * bbox_h), 0, label.shape[0] - 1)
        hmax = np.clip((hmax + bbox_enlarge * bbox_h), hmin + 1, label.shape[0])

        bbox[i, :] = [int(wmin), int(hmin), int(wmax), int(hmax)]

    return bbox.astype(int)


def label_to_prob(label, channels):
    prob = np.zeros(label.shape + (channels * 2, ))
    for i in range(channels):
        prob[(label == i + 1), i * 2 + 1] = 1
        prob[(label != i + 1), i * 2] = 1
    return prob


def combine_prob(prob):
    temp_prob = np.zeros(prob.shape[0:2] + (prob.shape[2] // 2 + 1, ))
    temp_prob[..., 0] = 1
    for i in range(1, temp_prob.shape[2]):
        temp_prob[..., i] = prob[..., i * 2 - 1]
        temp_prob[..., 0] *= prob[..., i * 2 - 2]

    temp_prob = temp_prob / np.sum(temp_prob, axis=2)[..., np.newaxis]
    return temp_prob


def prob_to_label(prob):
    label = np.argmax(prob, axis=2)
    return label


def IoU(bbox1, bbox2):
    s1 = max(0, (min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1]) + 1)) * max(0, (min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0]) + 1))
    s2 = (max(bbox1[3], bbox2[3]) - min(bbox1[1], bbox2[1] + 1)) * (max(bbox1[2], bbox2[2]) - min(bbox1[0], bbox2[0] + 1))
    return float(s1) / float(s2)
