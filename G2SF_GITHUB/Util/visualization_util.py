import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib
from .util import de_normalizer
import open3d as o3d

__all__ = ['plot_fig', 'plot_pcd']


def remake_format(figures):
    new_figures, info = list(), list()
    for fig in figures:
        if type(fig) is not np.ndarray:
            fig = fig.numpy()
        if len(fig.shape) == 3:
            if fig.shape[0] == 3:
                fig = de_normalizer(fig)
                new_figures.append(fig)
                info.append('rgb')
            else:
                new_figures.append(fig[0])
                info.append('heat_map')
        else:
            new_figures.append(fig)
            info.append('binary')
    return new_figures, info


""" Figures: list([3, w, h] or [1, w, h] or [w, h]) (rgb, heat_map, binary)
"""


def plot_fig(figures, labels, title='Untitled', show=True, save_name=None, save_path=None):
    figures, info = remake_format(figures)
    fig, axs = plt.subplots(1, len(figures), figsize=(6 * len(figures), 6))

    for ax in axs:
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
    for i, subfig in enumerate(figures):
        if info[i] == 'rgb':
            subfig[subfig <= 0] = 0
            subfig[subfig >= 255] = 255
            axs[i].imshow(subfig)
        elif info[i] == 'binary':
            subfig[subfig <= 0] = 0
            subfig[subfig >= 255] = 255
            axs[i].imshow(subfig, cmap='gray')
        else:
            # axs[i].imshow(subfig, cmap='jet', alpha=0.5, interpolation='none', vmin=0, vmax=1)
            axs[i].imshow(subfig, cmap='jet', interpolation='none')
        axs[i].title.set_text(labels[i])

    fig.suptitle(title)
    fig.tight_layout()
    if save_name is not None and save_path is not None:
        plt.savefig(os.path.join(save_path, save_name), dpi=128)
        plt.clf()
    if show:
        plt.show()
    plt.close()


def plot_pcd(pcd):
    if type(pcd) is not np.ndarray:
        pcd = pcd.numpy()
    if len(pcd.shape) == 3:
        pcd = pcd.transpose(1, 2, 0)
        pcd = pcd.reshape(-1, 3)

    nonzero_indices = np.nonzero(np.all(pcd != 0, axis=-1))[0]
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd[nonzero_indices])
    o3d.visualization.draw_geometries([pcd_o3d])



