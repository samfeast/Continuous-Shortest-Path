import matplotlib.pyplot as plt
import numpy as np

import matplotlib.patches as patches
from matplotlib.path import Path


def draw(g,ps):
    fig,ax = plt.subplots()
    ax.set_xlim(0, g.get_target()[0])
    ax.set_ylim(g.get_target()[1],0)
    for shape in g.get_regions()[:-4]:
        codes=[Path.MOVETO]
        shape_1=shape[:-1]
        for i in range(len(shape_1)):
            codes.append(Path.LINETO)
        shape_1.append(shape_1[0])
        path=Path(shape_1,codes)
        patch = patches.PathPatch(path, facecolor='None', lw=1)
        ax.add_patch(patch)
    for p in ps:
        codes=[Path.MOVETO]
        for i in range(len(p)-1):
            codes.append(Path.LINETO)
        path=Path(p,codes)
        patch = patches.PathPatch(path, facecolor='None',edgecolor='r', lw=1)
        ax.add_patch(patch)
    plt.show()

