from itertools import product

import pandas as pd

from tensorflow.keras.models import load_model

from modules.utils.data_utils import embedding_extraction
from modules.utils.general_utils import dirs_creation
from modules.dimensionality_reduction import UMAP_tuning

PROJECT_NAME = 'books'
dirs_creation(
    [f'results\\figures\\{PROJECT_NAME}'],
    wipe_dir=True
)
TARGET_DECODER = pd.read_pickle(
    f'results\\objects\\{PROJECT_NAME}\\target_decoder.pkl'
)
N_NEIGHBORS = [5, 15, 45, 135, 400]
MIN_DIST = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9]

targets_themes = {
    'A Clockwork Orange': 'plasma',
    'The Man Who Mistook His Wife for a Hat': 'winter',
    "Alice's Adventures in Wonderland": 'viridis',
    'Trainspotting': 'magma',
    "The Hitchiker's Guide to the Galaxy": 'summer',
    'The Little Prince': 'spring',
    'The Handmaids Tale': 'autumn',
    'Girl, Woman, Other': 'cool',
}
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
