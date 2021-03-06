from tqdm import tqdm

import numpy as np

from umap import UMAP

import matplotlib.pyplot as plt

from modules.utils.general_utils import dirs_creation


def UMAP_fitting(array, n_components, n_neighbors, min_dist, **kwargs):
    """
    """
    reduction = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        **kwargs
    ).fit_transform(array)
    return reduction


def UMAP_tuning(array, targets, colors, parameters_combination, targets_themes,
                figsize, project_name, n_components=2, embed_targets=False,
                **kwargs):
    """
    """
    root = f'results\\umap_tune\\{project_name}'
    dirs_creation(
        [f'{root}\\{para[0]}_{para[1]}' for para in parameters_combination],
        wipe_dir=True
    )
    plt.style.use('dark_background')
    for parameters in tqdm(parameters_combination):

        reduction = UMAP_fitting(
            array=array,
            n_components=2,
            n_neighbors=parameters[0],
            min_dist=parameters[1],
            **kwargs
        )

        if embed_targets:
            for target, theme in targets_themes.items():

                index = np.argwhere(targets == target).flatten()
                target_reduction = UMAP_fitting(
                    array=array[index],
                    n_components=2,
                    n_neighbors=parameters[0],
                    min_dist=parameters[1],
                    **kwargs
                )
                fig_target, ax_target = plt.subplots(figsize=(10, 10))
                ax_target.scatter(
                    target_reduction[:, 0],
                    target_reduction[:, 1],
                    s=0.25,
                    c=colors[index],
                    cmap=theme,
                    edgecolor='',
                    marker='o'
                )
                ax_target.axis('off')

                ax_target.text(
                    0.5,
                    1,
                    target.upper(),
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax_target.transAxes,
                    fontname='Microsoft Yi Baiti',
                    size=20,
                    weight='bold'
                )

                fig_target.savefig(
                    f'{root}\\{parameters[0]}_{parameters[1]}\\{target}_e.png',
                    dpi=400
                )

        fig_main, ax_main = plt.subplots(figsize=figsize)
        for target, theme in targets_themes.items():

            index = np.argwhere(targets == target).flatten()
            fig_sub, ax_sub = plt.subplots(figsize=(10, 10))
            ax_sub.scatter(
                reduction[:, 0][index],
                reduction[:, 1][index],
                s=0.25,
                c=colors[index],
                cmap=theme,
                edgecolor='',
                marker='o'
            )
            ax_sub.axis('off')

            ax_sub.text(
                0.5,
                1,
                target.upper(),
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax_sub.transAxes,
                fontname='Microsoft Yi Baiti',
                size=20,
                weight='bold'
            )
            fig_sub.savefig(
                f'{root}\\{parameters[0]}_{parameters[1]}\\{target}.png',
                dpi=400
            )

            ax_main.scatter(
                reduction[:, 0][index],
                reduction[:, 1][index],
                s=0.25,
                c=colors[index],
                cmap=theme,
                edgecolor='',
                marker='o'
            )

            ax_main.axis('off')

        ax_main.text(
            0.5,
            1,
            'BOOKS GALAXY',
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax_main.transAxes,
            fontname='Microsoft Yi Baiti',
            size=20,
            weight='bold'
        )
        fig_main.savefig(
            f'{root}\\{parameters[0]}_{parameters[1]}\\galaxy.png',
            dpi=400
        )
        plt.close('all')
