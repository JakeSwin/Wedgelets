import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')

from PIL import Image
from pykrige.ok import OrdinaryKriging

from quadtree import QuadTree

CROP_SIZE = 150
NUM_SAMPLES = 2000
IMG_WIDTH = 0
IMG_HEIGHT = 0

def sample_orthomosaic(image):
    x = np.random.randint(low=CROP_SIZE, high=IMG_WIDTH-CROP_SIZE+1, size=NUM_SAMPLES)
    y = np.random.randint(low=CROP_SIZE, high=IMG_HEIGHT-CROP_SIZE+1, size=NUM_SAMPLES)
    weed_chance = np.zeros(NUM_SAMPLES)

    for i in range(NUM_SAMPLES):
        crop = image.crop((x[i], y[i], x[i]+CROP_SIZE, y[i]+CROP_SIZE))
        weed_chance[i] = np.count_nonzero(np.array(crop)[:, :, 0]) / CROP_SIZE**2
    
    return x, y, weed_chance

def sample_GP(KrigingObj):
    gridx = np.arange(0, IMG_WIDTH, 22, dtype='float64')
    gridy = np.arange(0, IMG_HEIGHT, 22, dtype='float64')
    zstar, ss = OK.execute("grid", gridx, gridy)
    return zstar, ss

if __name__ == "__main__":
    im = Image.open("000/000/groundtruth/first000_gt.png")
    IMG_WIDTH, IMG_HEIGHT = im.size

    x, y, weed_chance = sample_orthomosaic(im)

    OK = OrdinaryKriging(
        x,
        y,
        weed_chance,
        variogram_model='exponential',
        verbose=True,
    )

    zstar, ss = sample_GP(OK)
    normalised = (zstar.data - np.min(zstar.data)) / (np.max(zstar.data) - np.min(zstar.data))
    # imgplot = plt.imshow(normalised)
    # plt.show()

    quadtree = QuadTree(normalised)

    depth = 6
    image = quadtree.create_image(depth, show_lines=False)

    quadtree.create_gif("zstar_quadtree.gif", show_lines=True)
    image.save("zstar_quadtree.png")
