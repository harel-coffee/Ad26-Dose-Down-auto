# Daniel Zhu
# October 10th, 2021
# General functions to help in plotting things.
# Import pretty much everything I know that's involved in creating plots:
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly

import numpy as np


def shifted_colormap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the median value of a colormap and scale the remaining color range.
    :param cmap: The colormap to be altered.
    :param start: Offset from the lowest point in the original colormap's range, given as a fraction of the total
    range. Should be between 0 and 0.5; defaults to 0 for no lower offset.
    :param midpoint: The new center of the colormap, given as a fraction of the total range. Should be between 0 and
    1.0, defaulting to 0.5 for no shift. Usually the optimal value is abs(vmin)/(vmax+abs(vmin)).
    :param stop: Offset from the highest point in the original colormap's range, given as a fraction of the total
    range. Should be between 0.5 and 1; defaults to 1 for no offset.
    :param name: Name of the new color map.
    :return: The new colormap object.
    '''
    # If colormap is given as a string, retrieve the colormap object:
    if isinstance(cmap, str):
        cmap = matplotlib.cm.get_cmap(cmap)

    cmap_dict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # Regular index to assign colors to range based on the chosen starting and stopping points:
    reg_index = np.hstack([np.linspace(start, 0.5, 128, endpoint=False), np.linspace(0.5, stop, 129)])

    # Shifted index to indicate where along the regular index colors should be placed.
    shift_index = np.hstack([np.linspace(0.0, midpoint, 128, endpoint=False), np.linspace(midpoint, 1.0, 129)])

    # Scale the offset color bar:
    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        # Creating colormaps: three item tuple: first element can be thought of as a location along the colorbar,
        # and the second and third give the color at that location. In the vast majority of cases, these two elements
        # will be identical in value.
        cmap_dict['red'].append((si, r, r))
        cmap_dict['green'].append((si, g, g))
        cmap_dict['blue'].append((si, b, b))
        cmap_dict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cmap_dict)
    plt.register_cmap(cmap=newcmap)
    return newcmap