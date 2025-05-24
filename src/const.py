import numpy as np

CLASSES = [
    "helmet",
    "no_helmet",
    "rider",
    "motorcycle"
]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
