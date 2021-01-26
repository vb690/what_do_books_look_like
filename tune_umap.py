from itertools import product

import pandas as pd

from tensorflow.keras.models import load_model

from modules.utils.data_utils import embedding_extraction
from modules.utils.general_utils import dirs_creation
from modules.dimensionality_reduction import UMAP_tuning


def run_tune_umap(PROJECT_NAME, N_NEIGHBORS=None, MIN_DIST=None, **kwargs):
    """
    """
    dirs_creation(
        [f'results\\umap_tune\\{PROJECT_NAME}'],
        wipe_dir=True
    )
    TARGET_DECODER = pd.read_pickle(
        f'results\\objects\\{PROJECT_NAME}\\target_decoder.pkl'
    )
    if N_NEIGHBORS is None:
        N_NEIGHBORS = [5, 15, 45, 135, 400]
    if MIN_DIST is None:
        MIN_DIST = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9]

    targets_themes = pd.read_pickle(
        f'results\\objects\\{PROJECT_NAME}\\themes.pkl'
    )
    parameters_combination = list(product(N_NEIGHBORS, MIN_DIST))

    model = load_model(f'results\\models\\{PROJECT_NAME}')

    encodes, targets, colors = embedding_extraction(
        model=model,
        project_name=PROJECT_NAME,
        target_decoder=TARGET_DECODER,
        colors_bins=20,
        extraction_point='features_extractor'
    )

    UMAP_tuning(
        array=encodes,
        targets=targets,
        colors=colors,
        parameters_combination=parameters_combination,
        targets_themes=targets_themes,
        figsize=(10, 10),
        n_components=2,
        project_name=PROJECT_NAME,
        embed_targets=True,
        **kwargs
    )


if __name__ == '__main__':
    PROJECT_NAME = input('Provide project name: ')
    run_tune_umap(
        PROJECT_NAME=PROJECT_NAME,
        N_NEIGHBORS=[5],
        MIN_DIST=[0.2],
        metric='cosine',
        verbose=True,
        n_epochs=1000
    )
