import math

import cv2
import numpy as np
import pyquaternion as pq


def getJetColor(v, vmin, vmax):
    c = np.zeros((3))
    if (v < vmin):
        v = vmin
    if (v > vmax):
        v = vmax
    dv = vmax - vmin
    if (v < (vmin + 0.125 * dv)):
        c[0] = 256 * (0.5 + (v * 4))  # B: 0.5 ~ 1
    elif (v < (vmin + 0.375 * dv)):
        c[0] = 255
        c[1] = 256 * (v - 0.125) * 4  # G: 0 ~ 1
    elif (v < (vmin + 0.625 * dv)):
        c[0] = 256 * (-4 * v + 2.5)  # B: 1 ~ 0
        c[1] = 255
        c[2] = 256 * (4 * (v - 0.375))  # R: 0 ~ 1
    elif (v < (vmin + 0.875 * dv)):
        c[1] = 256 * (-4 * v + 3.5)  # G: 1 ~ 0
        c[2] = 255
    else:
        c[2] = 256 * (-4 * v + 4.5)  # R: 1 ~ 0.5
    return c


def colorize(gray_img):
    out = np.zeros(gray_img.shape + (3,))
    for y in range(out.shape[0]):
        for x in range(out.shape[1]):
            out[y, x, :] = getJetColor(gray_img[y, x], 0, 1)
    return out


def padRightDownCorner(img):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0  # up
    pad[1] = 0  # left
    pad[2] = 0 if (h % 8 == 0) else 8 - (h % 8)  # down
    pad[3] = 0 if (w % 8 == 0) else 8 - (w % 8)  # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1, :, :] * 0 + 128, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:, 0:1, :] * + 128, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1, :, :] * + 128, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:, -2:-1, :] * 0 + 128, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad


def map_resize(new_shape, heatmap):
    # Resizes the output back to the size of the test image
    scale_y = new_shape[0] / float(heatmap.shape[0])
    scale_x = new_shape[1] / float(heatmap.shape[1])
    map_resized = cv2.resize(heatmap, None, fx=scale_x, fy=scale_y,
                             interpolation=cv2.INTER_CUBIC)

    return map_resized


def get_quaternion(p, q, orientation="front"):
    dx, dy, dz = p - q
    dx = -dx
    dy = -dy

    available_orientations = ["front", "back"]
    if orientation not in available_orientations:
        print("WARNING: '%s' orientation not valid! Assuming 'front")

    if orientation == "back":
        yaw = pq.Quaternion(axis=[0, 1, 0], degrees=-math.pi)
    else:
        yaw = pq.Quaternion(axis=[0, 1, 0], degrees=0)


    if dz > 0:
        roll = pq.Quaternion(axis=[0, 0, 1], degrees=-math.atan2(dx, dz))
        pitch = pq.Quaternion(axis=[1, 0, 0], degrees=-math.atan2(dy, dz))
    else:
        roll = pq.Quaternion(axis=[0, 0, 1], degrees=math.pi + math.atan2(dx, dz))
        yaw = pq.Quaternion(axis=[0, 1, 0], degrees=-math.pi)
        pitch = pq.Quaternion(axis=[1, 0, 0], degrees=-math.atan2(dy, dz))

    return roll * yaw * pitch
