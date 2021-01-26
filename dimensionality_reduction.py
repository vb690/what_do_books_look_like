import numpy as np

import pandas as pd

from tensorflow.keras.models import load_model

from umap import UMAP

from modules.utils.general_utils import dirs_creation
from modules.utils.data_utils import embedding_extraction


def run_dimensionality_reduction(PROJECT_NAME, LOCAL_N_NEIGH=5,
                                 LOCAL_MIN_DIST=0.2, GLOBAL_N_NEIGH=5,
                                 GLOBAL_MIN_DIST=0.2, **kwargs):
    """
    """
    TARGET_DECODER = pd.read_pickle(
        f'results\\objects\\{PROJECT_NAME}\\target_decoder.pkl'
    )

    model = load_model(f'results\\models\\{PROJECT_NAME}')

    encodes, targets_array, colors = embedding_extraction(
        model=model,
        project_name=PROJECT_NAME,
        target_decoder=TARGET_DECODER,
        colors_bins=20,
        extraction_point='features_extractor'
        )

    for dims in [2, 3]:

        root = f'results\\objects\\{PROJECT_NAME}\\{dims}D'
        dirs_creation(
            [root],
            wipe_dir=True
        )

        print(f'Reduction for {dims}D galaxy.')
        reduction = UMAP(
                n_components=dims,
                n_neighbors=GLOBAL_N_NEIGH,
                min_dist=GLOBAL_MIN_DIST,
                **kwargs
        ).fit_transform(encodes)

        np.save(
            f'results\\objects\\{PROJECT_NAME}\\{dims}D\\galaxy',
            reduction
        )

        for unique_target in np.unique(targets_array):

            index = np.argwhere(targets_array == unique_target).flatten()

            print(f'Reduction for {dims}D {unique_target}.')
            reduction = UMAP(
                n_components=dims,
                n_neighbors=LOCAL_N_NEIGH,
                min_dist=LOCAL_MIN_DIST,
                verbose=True,
                n_epochs=1000
            ).fit_transform(encodes[index])

            np.save(
                f'results\\objects\\{PROJECT_NAME}\\{dims}D\\{unique_target}',
                reduction
            )


if __name__ == '__main__':
    PROJECT_NAME = input('Provide project name: ')
    run_dimensionality_reduction(
        PROJECT_NAME=PROJECT_NAME,
        n_epochs=1000,
        verbose=True,
        metric='cosine'
    )
