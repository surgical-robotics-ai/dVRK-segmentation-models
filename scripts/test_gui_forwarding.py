import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import numpy as np


path = "/home/jbarrag3/research_juan/data/phantom2_3d_med_rec/rec01/raw/img_000000.png"


im = np.array(Image.open(path))

fig, ax = plt.subplots(1)
ax.imshow(im)
plt.show()
