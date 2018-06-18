import numpy as np


def readFlow(name):
    f = open(name, 'rb')

    header = f.read(4)
    if header.decode("utf-8") != 'PIEH':
        raise Exception('Flow file header does not contain PIEH')

    width = np.fromfile(f, np.int32, 1).squeeze()
    height = np.fromfile(f, np.int32, 1).squeeze()

    flow = np.fromfile(f, np.float32, width * height * 2).reshape((height,
                                                                   width, 2))
    f.close()
    return flow.astype(np.float32)


def get_warp_label(flow1, flow2, label1):
    label2 = np.zeros_like(label1, dtype=label1.dtype)
    height = flow1.shape[0]
    width = flow1.shape[1]
    flow_t = np.zeros_like(flow1, dtype=flow1.dtype)

    grid = np.indices((height, width)).swapaxes(0, 1).swapaxes(1, 2)
    dx = grid[:, :, 0] + flow2[:, :, 1]
    dy = grid[:, :, 1] + flow2[:, :, 0]
    sx = np.floor(dx).astype(int)
    sy = np.floor(dy).astype(int)
    valid = (sx >= 0) & (sx < height - 1) & (sy >= 0) & (sy < width - 1)

    sx_mat = np.dstack((sx, sx + 1, sx, sx + 1)).clip(0, height - 1)
    sy_mat = np.dstack((sy, sy, sy + 1, sy + 1)).clip(0, width - 1)
    sxsy_mat = np.abs((1 - np.abs(sx_mat - dx[:, :, np.newaxis])) *
                      (1 - np.abs(sy_mat - dy[:, :, np.newaxis])))

    for i in range(4):
        flow_t = flow_t + sxsy_mat[:, :, i][:, :, np.
                                            newaxis] * flow1[sx_mat[:, :, i],
                                                             sy_mat[:, :, i], :]

    valid = valid & (np.linalg.norm(
        flow_t[:, :, [1, 0]] + np.dstack((dx, dy)) - grid, axis=2) < 100)

    flow_t = (flow2 - flow_t) / 2.0
    dx = grid[:, :, 0] + flow_t[:, :, 1]
    dy = grid[:, :, 1] + flow_t[:, :, 0]

    valid = valid & (dx >= 0) & (dx < height - 1) & (dy >= 0) & (dy < width - 1)
    label2[valid, :] = label1[dx[valid].round().astype(int), dy[valid].round()
                              .astype(int), :]
    return label2
