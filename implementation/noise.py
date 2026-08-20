import numpy as np

from constants import eta, field, q


def generate_noise(shape: tuple) -> np.ndarray:
    """
    Using the "centered binomial distribution".
    The generation is UNSAFE. This is just a prototype.
    """

    noise = np.zeros(shape, dtype=np.int64)
    for i in range(eta):
        a = np.random.randint(2, size=shape)
        b = np.random.randint(2, size=shape)
        noise += a - b
    return field(noise % q)
