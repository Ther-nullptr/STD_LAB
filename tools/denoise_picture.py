import numpy as np

def box_kernel(size):
    K = np.ones((size, size))
    # your code
    K = K / (size * size)
    return K


def gaussian_kernel(size, sigma):
    K = np.zeros((size, size))
    pad = size // 2

    for x in range(-pad, -pad + size):
        for y in range(-pad, -pad + size):
            K[x + size // 2][y + size // 2] = np.exp(-(x**2 + y**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)

    K /= K.sum()
    return K


def img_box_filter(image, r):
    K = box_kernel(r)
    L, W = image.shape  # L: length; W: width

    im_pad = np.zeros((L + r - 1, W + r - 1))
    im_pad[r // 2 : L + r // 2, r // 2 : W + r // 2] = image
    im_output = np.zeros((L, W))

    for i in range(r // 2, L + r // 2):  # length position
        for j in range(r // 2, W + r // 2):  # width position
            im_output[i - r // 2][j - r // 2] = np.sum(im_pad[i - r // 2: i + r // 2 + 1, j - r // 2: j + r // 2 + 1] * K)

    return im_output


def img_gaussian_filter(image, r, sigma):
    K = gaussian_kernel(r, sigma)
    L, W = image.shape  

    im_pad = np.zeros((L + r - 1, W + r - 1))
    im_pad[r // 2 : L + r // 2, r // 2 : W + r // 2] = image
    im_output = np.zeros((L, W))

    for i in range(r // 2, L + r // 2):
        for j in range(r // 2, W + r // 2):  
            im_output[i - r // 2][j - r // 2] = np.sum(im_pad[i - r // 2: i + r // 2 + 1, j - r // 2: j + r // 2 + 1] * K)

    return im_output



def img_median_filter(image, r):
    L, W = image.shape  # L: length; W: width; C: channel

    im_pad = np.zeros((L + r - 1, W + r - 1))
    im_pad[r // 2:L + r // 2, r // 2:W + r // 2] = image
    im_output = np.zeros((L, W))

    for i in range(r // 2, L + r // 2):  # length position
        for j in range(r // 2, W + r // 2):  # width position
            im_output[i - r // 2][j - r // 2] = np.median(im_pad[i - r // 2:i + r // 2 + 1, j - r // 2:j + r // 2 + 1])

    return im_output