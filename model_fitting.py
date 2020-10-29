import os

import numpy as np

import pandas as pd

from tensorflow.keras.callbacks import EarlyStopping
from kerastuner.tuners import Hyperband

from modules.models import LanguageModel
from modules.utils.data_utils import DataGenerator

SENTENCE_DECODER = pd.read_pickle(
    'results\\objects\\books\\sentence_decoder.pkl'
)
TARGET_DECODER = pd.read_pickle(
    'results\\objects\\books\\target_decoder.pkl'
)

PROJECT_NAME = 'books'
TUNING_FRACTION = 0.1

BTCH_LIST = os.listdir(f'data\\preprocessed\\{PROJECT_NAME}\\inputs')
BTCH = [i for i in range(len(BTCH_LIST))]
BTCH = np.random.choice(BTCH, int(len(BTCH) * TUNING_FRACTION), replace=False)

TR_BTCH = BTCH[: int(len(BTCH) * 0.8)]
TS_BTCH = BTCH[int(len(BTCH) * 0.8):]

tr_generator = DataGenerator(
    list_batches=TR_BTCH,
    project_name=PROJECT_NAME,
    shuffle=True,
    multi_target=True
)

ts_generator = DataGenerator(
    list_batches=TS_BTCH,
    project_name=PROJECT_NAME,
    shuffle=True,
    multi_target=True
)

model = LanguageModel(
    max_vocab=len(SENTENCE_DECODER) + 1,
    multi_target=True,
    max_target_1=len(SENTENCE_DECODER) + 1,
    max_target_2=len(TARGET_DECODER)
)

ES = EarlyStopping(
    monitor='val_loss',
    min_delta=0.0001,
    patience=5,
    verbose=1,
    mode='auto',
    restore_best_weights=True
)

tuner_obj = Hyperband(
    hypermodel=model,
    max_epochs=30,
    hyperband_iterations=1,
    objective='val_loss',
    directory='o',
    project_name=f'{PROJECT_NAME}'
)

tuner_obj.search(
    tr_generator,
    epochs=30,
    callbacks=[ES],
    verbose=2,
    validation_data=ts_generator
)

BTCH = [i for i in range(len(BTCH_LIST))]
BTCH = np.random.choice(BTCH, int(len(BTCH)), replace=False)

TR_BTCH = BTCH[: int(len(BTCH) * 0.8)]
TS_BTCH = BTCH[int(len(BTCH) * 0.8):]

tr_generator = DataGenerator(
    list_batches=TR_BTCH,
    project_name=PROJECT_NAME,
    shuffle=True,
    multi_target=True
)

ts_generator = DataGenerator(
    list_batches=TS_BTCH,
    project_name=PROJECT_NAME,
    shuffle=True,
    multi_target=True
)

model = tuner_obj.get_best_models(1)[0]
model.fit(
    tr_generator,
    epochs=30,
    verbose=2,
    callbacks=[ES],
    validation_data=ts_generator
)
model.save(f'results\\models\\{PROJECT_NAME}')
