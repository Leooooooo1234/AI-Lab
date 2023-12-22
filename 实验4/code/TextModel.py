from keras import layers, models
class TextModel(object):
    def __init__(
            self, max_sequence_length, lens):
        self.max_sequence_length = max_sequence_length
        self.lens = lens

    def build_GRU_model(self):
        encoder_input = layers.Input(shape=(self.max_sequence_length,))
        encoder_embedding = layers.Embedding(self.lens, 256)(encoder_input)
        encoder_gru = layers.GRU(units=256, return_state=True)
        encoder_outputs, state_h = encoder_gru(encoder_embedding)
        encoder_states = [state_h]

        decoder_input = layers.Input(shape=(self.max_sequence_length,))
        decoder_embedding = layers.Embedding(self.lens, 256)(decoder_input)
        decoder_gru = layers.GRU(units=256, return_sequences=True, return_state=True)
        decoder_outputs, _ = decoder_gru(decoder_embedding, initial_state=encoder_states)
        decoder_dense = layers.Dense(self.lens, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        model = models.Model([encoder_input, decoder_input], decoder_outputs)

        return model



    def build_LSTM_model(self):
        encoder_input = layers.Input(shape=(self.max_sequence_length,))
        encoder_embedding = layers.Embedding(self.lens, 256)(encoder_input)
        encoder_lstm = layers.LSTM(units=256, return_state=True)
        encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
        encoder_states = [state_h, state_c]

        decoder_input = layers.Input(shape=(self.max_sequence_length,))
        decoder_embedding = layers.Embedding(self.lens, 256)(decoder_input)
        decoder_lstm = layers.LSTM(units=256, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
        decoder_dense = layers.Dense(self.lens, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        model = models.Model([encoder_input, decoder_input], decoder_outputs)

        return model





