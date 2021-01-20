import os

import re

from tqdm import tqdm

import numpy as np

import pdfplumber

import pandas as pd

from skimage.exposure import equalize_hist

from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence


class DataGenerator(Sequence):
    """
    Class implementing a data generator
    """
    def __init__(self, list_batches, project_name, shuffle=True,
                 multi_target=False):
        """
        """
        self.list_batches = list_batches
        self.shuffle = shuffle
        self.multi_target = multi_target
        self.root_dir = f'data\\preprocessed\\{project_name}'
        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch'
        """
        return int(len(self.list_batches))

    def __getitem__(self, index):
        """
        Generate one batch of data
        """
        # Pick a batch
        batch = self.list_batches[index]
        # Generate X and y
        X, y = self.__data_generation(batch)
        return X, y

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        if self.shuffle is True:
            np.random.shuffle(self.list_batches)

    def __data_generation(self, batch):
        """Generates data containing batch_size samples
        """
        X_sentence = np.load(
            f'{self.root_dir}\\inputs\\{batch}.npy',
            allow_pickle=True
        )
        rows, cols = X_sentence.shape
        X_order = np.array([[i for i in range(cols)] for j in range(rows)])
        X_order = X_order.reshape((rows, cols, 1))
        if self.multi_target:
            y_sentence = np.load(
                f'{self.root_dir}\\targets\\{batch}_1.npy',
                allow_pickle=True
            )
            y_class = np.load(
                f'{self.root_dir}\\targets\\{batch}_2.npy',
                allow_pickle=True
            )
            return [X_sentence, X_order], [y_sentence, y_class]
        else:
            y_sentence = np.load(
                f'{self.root_dir}\\targets\\{batch}_1.npy',
                allow_pickle=True
            )
            return [X_sentence, X_order], y_sentence


def pdf_plumbering(pdfs_path, project_id):
    """
    """
    regex = re.compile(r'[\n\r\t]')
    list_pdfs = []
    for name in os.listdir(f'{pdfs_path}\\{project_id}'):

        print(f'Plumbering {name}')
        total_text = ''
        with pdfplumber.open(f'{pdfs_path}\\{project_id}\\{name}') as pdf:
            for page in tqdm(pdf.pages):
                txt = page.extract_text()
                if txt is None:
                    continue
                replaced = regex.sub(' ', txt)
                total_text += replaced

        sentences = np.array(total_text.split('.'))
        df = pd.DataFrame(columns=['sentences'])
        df['sentences'] = sentences
        df['document'] = name
        df = df.dropna()
        list_pdfs.append(df)

    df = pd.concat(list_pdfs)
    df.to_csv(f'data\\preprocessed\\{project_id}\\{project_id}_df.csv')
    return df


def tokenization(sentence):
    """
    """
    tokens = sentence.split()
    tokens = [token.lower() for token in tokens]
    tokens = ['<GO>'] + tokens + ['<EOS>']
    return tokens


def encoding(bag_of_categories, nested=False, start=0):
    """
    """
    if nested:
        unique_categories = set(
            [cat for list_cat in bag_of_categories for cat in list_cat]
        )
    else:
        unique_categories = set(bag_of_categories)

    encoder = {cat: code for code, cat in enumerate(unique_categories, start)}
    decoder = {code: cat for cat, code in encoder.items()}
    return encoder, decoder


def preprocessing(list_sentences, targets, project_id, max_len=40,
                  max_batch=64):
    """
    """
    # create empty dictionaries
    dict_sentences = {length: [] for length in range(3, max_len + 1)}
    dict_targets = {length: [] for length in range(3, max_len + 1)}

    # tokenize the sentences
    bag_of_sentences = [
        tokenization(sentence) for sentence in list_sentences
    ]

    # create encoding
    sentence_encoder, sentence_decoder = encoding(
        bag_of_categories=bag_of_sentences,
        nested=True
    )
    target_encoder, target_decoder = encoding(
        bag_of_categories=targets,
        nested=False
    )
    for sentence, target in zip(bag_of_sentences, targets):

        if len(sentence) < 3:
            continue
        bag_of_words = [sentence_encoder[word] for word in sentence]
        if len(bag_of_words) > max_len:
            for cut in range(0, len(bag_of_words), max_len):

                trim_bag_of_words = bag_of_words[cut:cut + max_len]
                if len(trim_bag_of_words) < 3:
                    continue
                dict_sentences[
                    len(trim_bag_of_words)].append(trim_bag_of_words)

                trim_bag_of_targets = [
                    target_encoder[target]] * len(trim_bag_of_words)
                dict_targets[
                    len(trim_bag_of_targets)].append(trim_bag_of_targets)
        else:
            dict_sentences[len(bag_of_words)].append(bag_of_words)

            bag_of_targets = [target_encoder[target]] * len(bag_of_words)
            dict_targets[len(bag_of_words)].append(bag_of_targets)

    batch_count = 0
    for length in range(3, max_len):

        sentence_batch = np.array(dict_sentences[length])

        if len(sentence_batch.shape) < 2:
            continue
        sentence_tar_batch = sentence_batch[:, 1:]
        sentence_tar_batch = sentence_tar_batch.reshape(
            (
                sentence_tar_batch.shape[0],
                sentence_tar_batch.shape[1],
                1
             )
        )
        sentence_batch = sentence_batch[:, :-1]

        class_tar_batch = np.array(dict_targets[length])
        class_tar_batch = class_tar_batch[:, 1:]
        class_tar_batch = class_tar_batch.reshape(
            (
                class_tar_batch.shape[0],
                class_tar_batch.shape[1],
                1
             )
        )
        num_batches = (
            sentence_batch.shape[0] + max_batch - 1
        ) // max_batch

        for batch_index in range(num_batches):

            minimum = min(
                sentence_batch.shape[0],
                (batch_index + 1) * max_batch
            )

            sentence_sub_batch = sentence_batch[
                batch_index * max_batch: minimum
            ]
            sentence_sub_batch = np.float32(sentence_sub_batch)

            sentence_tar_sub_batch = sentence_tar_batch[
                batch_index * max_batch: minimum
            ]
            sentence_tar_sub_batch = np.float32(sentence_tar_sub_batch)

            class_tar_sub_batch = class_tar_batch[
                batch_index * max_batch: minimum
            ]
            class_tar_sub_batch = np.float32(class_tar_sub_batch)

            if len(sentence_sub_batch.shape) < 2:
                print(sentence_sub_batch.shape)

            np.save(
                f'data\\preprocessed\\{project_id}\\inputs\\{batch_count}',
                sentence_sub_batch
            )
            np.save(
                f'data\\preprocessed\\{project_id}\\targets\\{batch_count}_1',
                sentence_tar_sub_batch
            )
            np.save(
                f'data\\preprocessed\\{project_id}\\targets\\{batch_count}_2',
                class_tar_sub_batch
            )

            batch_count += 1

    return sentence_encoder, sentence_decoder, target_encoder, target_decoder


def embedding_extraction(model, project_name, target_decoder, colors_bins=20,
                         extraction_point='features_extractor'):
    """
    """
    inputs = []
    for inp in model.inputs:

        inputs.append(inp)

    out = model.get_layer(extraction_point)
    encoder = Model(inputs, out.output)

    list_encodes = []
    list_sequences = []
    list_targets = []
    batch_list = os.listdir(f'data\\preprocessed\\{project_name}\\inputs')
    for batch in tqdm(range(len(batch_list))):

        X_text = np.load(
            f'data\\preprocessed\\{project_name}\\inputs\\{batch}.npy'
        )
        y_class = np.load(
            f'data\\preprocessed\\{project_name}\\targets\\{batch}_2.npy'
        )
        rows, cols = X_text.shape
        X_sequences = np.array([[i for i in range(cols)] for j in range(rows)])
        X_sequences = X_sequences.reshape((rows, cols, 1))

        encodes = encoder.predict([X_text, X_sequences])
        encodes = encodes[:, 1:, :]
        rows, t_steps, cols = encodes.shape
        encodes = encodes.reshape(rows * t_steps, cols)

        X_sequences = X_sequences[:, 1:]
        sequences = X_sequences.reshape(rows * t_steps)
        list_encodes.append(encodes)
        list_sequences.append(sequences)

        y_class = y_class[:, 1:, :]
        list_targets.append(y_class.reshape(rows * t_steps))

    encodes = np.vstack(list_encodes)
    sequences = np.hstack(list_sequences)
    targets = np.hstack(list_targets)
    targets = np.array([target_decoder[target][:-4] for target in targets])
    colors = equalize_hist(sequences, nbins=20)

    return encodes, targets, colors
