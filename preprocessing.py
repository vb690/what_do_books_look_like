from modules.utils.general_utils import dirs_creation, dump_pickle
from modules.utils.data_utils import pdf_plumbering, preprocessing


for project_name in ['books', 'got']:

    dirs_creation(
        [
            f'data\\preprocessed\\{project_name}',
            f'results\\objects\\{project_name}'
        ],
        wipe_dir=True
    )

    df = pdf_plumbering(
        'data\\raw',
        project_name
    )

    sentences = list(df['sentences'].values)
    targets = df['document'].values

    dirs_creation(
        [
            f'data\\preprocessed\\{project_name}\\inputs',
            f'data\\preprocessed\\{project_name}\\targets'
        ],
        wipe_dir=True
    )

    sentence_encoder, sentence_decoder, target_encoder, \
        target_decoder = preprocessing(
            list_sentences=sentences,
            targets=targets,
            project_id=project_name,
            max_len=1000,
            max_batch=64
         )

    dump_pickle(
        objs=[
            sentence_encoder,
            sentence_decoder,
            target_encoder,
            target_decoder
        ],
        paths=[f'results\\objects\\{project_name}'] * 4,
        filenames=[
            'sentence_encoder',
            'sentence_decoder',
            'target_encoder',
            'target_decoder'
        ]
    )
