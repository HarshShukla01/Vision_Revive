# dehaze.py

import cv2
import numpy as np

def dark_channel(im, size=15):
    min_channel = np.min(im, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dark = cv2.erode(min_channel, kernel)
    return dark

def atm_light(im, dark):
    [h, w] = im.shape[:2]
    numpx = int(max(h * w / 1000, 1))
    darkvec = dark.reshape(h * w)
    imvec = im.reshape(h * w, 3)
    indices = darkvec.argsort()[-numpx:]
    atmsum = np.zeros([1, 3])
    for ind in indices:
        atmsum += imvec[ind]
    A = atmsum / numpx
    return A[0]

def transmission_estimate(im, A, omega=0.95, size=15):
    normed = im / A
    transmission = 1 - omega * dark_channel(normed, size)
    return transmission

def recover(im, t, A, tx=0.1):
    res = np.empty(im.shape, im.dtype)
    t = np.clip(t, tx, 1)[:, :, np.newaxis]
    res = (im - A) / t + A
    return np.clip(res, 0, 255).astype(np.uint8)

def dehaze(img):
    img = img.astype(np.float32)
    dark = dark_channel(img)
    A = atm_light(img, dark)
    te = transmission_estimate(img, A)
    result = recover(img, te, A)
    return result
