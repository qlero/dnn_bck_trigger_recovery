
import matplotlib.pyplot as plt
import numpy as np

def save_image(img, name, normalize=False):
    if normalize:
        img   = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(name)
