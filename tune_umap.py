from itertools import product

import pandas as pd

from tensorflow.keras.models import load_model

from modules.utils.data_utils import embedding_extraction
from modules.utils.general_utils import dirs_creation
from modules.dimensionality_reduction import UMAP_tuning


def run_tune_umap(PROJECT_NAME):
    """
    """
    dirs_creation(
        [f'results\\figures\\{PROJECT_NAME}'],
        wipe_dir=True
    )
    TARGET_DECODER = pd.read_pickle(
        f'results\\objects\\{PROJECT_NAME}\\target_decoder.pkl'
    )
    N_NEIGHBORS = [5, 15, 45, 135, 400]
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
        fraction=0.05,
        project_name='books',
        embed_targets=True
    )


if __name__ == '__main__':
    PROJECT_NAME = input('Provide project name: ')
    run_tune_umap(PROJECT_NAME=PROJECT_NAME)
