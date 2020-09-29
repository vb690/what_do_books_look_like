from tensorflow.keras.layers import Input, Concatenate, Activation
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.layers import SpatialDropout1D

from tensorflow.keras.models import Model

from kerastuner import HyperModel


class LanguageModel(HyperModel):
    '''
    '''
    def __init__(self, max_vocab, multi_target=False, max_target_1=None,
                 max_target_2=None):
        '''
        '''
        self.max_vocab = max_vocab
        self.multi_target = multi_target
        if multi_target:
            self.max_target_1 = max_target_1
            self.max_target_2 = max_target_2
        else:
            self.max_target = max_vocab

    def build(self, hp):
        '''
        '''
        dropout_rate = hp.Float(
            min_value=0.0,
            max_value=0.4,
            step=0.1,
            name='dropout_rate'
        )

        input_text = Input(
            shape=(None,),
            name='input_text'
        )
        input_seq = Input(
            shape=(None, 1),
            name='input_sequence'
        )

        # encoding
        embedding_text = Embedding(
            input_dim=self.max_vocab,
            output_dim=hp.Int(
                min_value=25,
                max_value=250,
                step=25,
                name='embedding_text'
            ),
            input_length=None,
            name='embedding_text'
        )(input_text)
        embedding_text = SpatialDropout1D(
            rate=dropout_rate,
            name='embedding_text_dropout'
        )(embedding_text)

        td_sequence = Dense(
            units=hp.Int(
                min_value=25,
                max_value=250,
                step=25,
                name='td_sequence'
                ),
            name='td_sequence'
        )(input_seq)
        td_sequence = SpatialDropout1D(
            rate=dropout_rate,
            name='td_sequence_dropout'
        )(td_sequence)

        text_features = Concatenate()([embedding_text, td_sequence])

        td_text_features = Dense(
            units=hp.Int(
                min_value=25,
                max_value=250,
                step=25,
                name='td_text_features'
                ),
            name='td_text_features'
        )(text_features)
        td_text_features = SpatialDropout1D(
            rate=dropout_rate,
            name='td_text_features_dropout'
        )(td_text_features)

        out_text = LSTM(
            units=hp.Int(
                min_value=25,
                max_value=250,
                step=25,
                name='lstm_text_features'
                ),
            return_sequences=True,
            name='features_extractor'
        )(td_text_features)
        out_text = SpatialDropout1D(
            rate=dropout_rate,
            name='out_text_dropout'
        )(out_text)

        if self.multi_target:

            # ################### FIRST TARGET ###################

            out_target_1 = Dense(
                units=hp.Int(
                    min_value=25,
                    max_value=250,
                    step=25,
                    name='td_out_target_1'
                    ),
                name='td_out_target_1'
            )(out_text)
            out_target_1 = SpatialDropout1D(
                rate=dropout_rate,
                name='td_out_target_1_dropout'
            )(out_target_1)

            out_target_1 = Activation('relu')(out_target_1)
            out_target_1 = Dense(
                units=self.max_target_1,
                name='out_target_1'
            )(out_target_1)
            out_target_1 = Activation('softmax')(out_target_1)

            # ################### SECOND TARGET ###################

            out_target_2 = Dense(
                units=hp.Int(
                    min_value=25,
                    max_value=250,
                    step=25,
                    name='td_out_target_2'
                    ),
                name='td_out_target_2'
            )(out_text)
            out_target_2 = SpatialDropout1D(
                rate=dropout_rate,
                name='td_out_target_2_dropout'
            )(out_target_2)

            out_target_2 = Activation('relu')(out_target_2)
            out_target_2 = Dense(
                units=self.max_target_2,
                name='out_target_2'
            )(out_target_2)
            out_target_2 = Activation('softmax')(out_target_2)

            outputs = [out_target_1, out_target_2]
            loss = ['sparse_categorical_crossentropy'] * 2

        else:
            out_target = Dense(
                units=hp.Int(
                    min_value=25,
                    max_value=250,
                    step=25,
                    name='td_out_target'
                    ),
                name='td_out_target'
            )(out_text)
            out_target = SpatialDropout1D(
                rate=dropout_rate,
                name='td_out_target_dropout'
            )(out_target)

            out_target = Activation('relu')(out_target)
            out_target = Dense(
                units=self.max_target,
                name='out_target'
            )(out_target)
            out_target = Activation('softmax')(out_target)

            outputs = out_target
            loss = ['sparse_categorical_crossentropy']

        model = Model(
            [input_text, input_seq],
            outputs
        )
        model.compile(
            loss=loss,
            optimizer='adam'
        )

        return model
