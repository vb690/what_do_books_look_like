import json

from modules.utils.general_utils import dirs_creation, dump_pickle
from modules.utils.data_utils import pdf_plumbering, preprocessing


def run_data_preprocessing(PROJECT_NAME, PROJECT_THEMES):
    """
    """
    dirs_creation(
        [
            f'data\\preprocessed\\{PROJECT_NAME}',
            f'results\\objects\\{PROJECT_NAME}'
        ],
        wipe_dir=True
    )

    df = pdf_plumbering(
        'data\\raw',
        PROJECT_NAME
    )

    sentences = list(df['sentences'].values)
    targets = df['document'].values

    dirs_creation(
        [
            f'data\\preprocessed\\{PROJECT_NAME}\\inputs',
            f'data\\preprocessed\\{PROJECT_NAME}\\targets'
        ],
        wipe_dir=True
    )

    sentence_encoder, sentence_decoder, target_encoder, \
        target_decoder = preprocessing(
            list_sentences=sentences,
            targets=targets,
            project_id=PROJECT_NAME,
            max_len=1000,
            max_batch=64
         )

    dump_pickle(
        objs=[
            sentence_encoder,
            sentence_decoder,
            target_encoder,
            target_decoder,
            PROJECT_THEMES
        ],
        paths=[f'results\\objects\\{PROJECT_NAME}'] * 5,
        filenames=[
            'sentence_encoder',
            'sentence_decoder',
            'target_encoder',
            'target_decoder',
            'themes'
        ]
    )


if __name__ == '__main__':
    PROJECT_NAME = input('Provide project name: ')
    PROJECT_THEMES = open(f'data\\jsons\\{PROJECT_NAME}.json')
    PROJECT_THEMES = json.load(PROJECT_THEMES)
    run_data_preprocessing(
        PROJECT_NAME=PROJECT_NAME,
        PROJECT_THEMES=PROJECT_THEMES
    )
