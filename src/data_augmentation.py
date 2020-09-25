import numpy as np


def gaussianNoise(img, mean=0, sigma=100):
    """
    Given a np.array img, this method returns a Gausian noised version of itself
    """

    img = img.copy().astype(float)
    noise = np.random.normal(mean, sigma, img.shape)
    mask_overflow_lower = img + noise < 0
    noise[mask_overflow_lower] = 0
    img += noise

    return img.astype(int)

