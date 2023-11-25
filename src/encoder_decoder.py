import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from attention import AttentionLayer


class Model:
    def __init__(self):
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        self.model = load_model('model.h5', custom_objects={'AttentionLayer': AttentionLayer})
        self.encoder_model = load_model('encoder_model.h5')
        self.decoder_model = load_model('decoder_model.h5', custom_objects={
                                        'AttentionLayer': AttentionLayer})

        with open('x_tokenizer', 'rb') as f:
            x_tokenizer = pickle.load(f)

        with open('y_tokenizer', 'rb') as f:
            y_tokenizer = pickle.load(f)

        self.reverse_target_word_index = y_tokenizer.index_word
        self.reverse_source_word_index = x_tokenizer.index_word
        self.target_word_index = y_tokenizer.word_index

    def seq2summary(self, output_seq):
        newString = ''

        for i in output_seq:
            if (i != 0 and i != self.target_word_index['sostok']) and i != self.target_word_index['eostok']:
                newString = newString + self.reverse_target_word_index[i] + ' '

        return newString

    def seq2text(self, input_seq):
        newString = ''

        for i in input_seq:
            if i != 0:
                newString = newString + self.reverse_source_word_index[i] + ' '

        return newString

    def beam_search_decoder(self, data, k=3):
        sequences = [[list(), 0.0]]

        for row in data:
            all_candidates = list()

            for i in range(len(sequences)):
                seq, score = sequences[i]

                for j in range(len(row)):
                    candidate = [seq + [j], score - np.log(row[j])]
                    all_candidates.append(candidate)

            ordered = sorted(all_candidates, key=lambda tup: tup[1])
            sequences = ordered[:k]

        return sequences

    def decode_sequence(self, input_seq):
        e_out, e_h, e_c = self.encoder_model.predict(input_seq)
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = self.target_word_index['sostok']
        stop_condition = False
        decoded_sentence = ''

        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict([target_seq] + [e_out, e_h, e_c])
            sampled_token_index = np.argmin(self.beam_search_decoder(output_tokens)[0][1])
            sampled_token = self.reverse_target_word_index[sampled_token_index]

            if(sampled_token != 'eostok'):
                decoded_sentence += ' ' + sampled_token

            if (sampled_token == 'eostok' or len(decoded_sentence.split()) >= 19):
                stop_condition = True

            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index
            e_h, e_c = h, c

        return decoded_sentence.strip()
