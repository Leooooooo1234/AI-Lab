import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras_preprocessing.sequence import pad_sequences
import os


seed = 666
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
def data_process(train_data, test_data):

    train_source_texts = train_data["description"]
    train_target_texts = train_data["diagnosis"]
    test_source_texts = test_data["description"]

    train_source_list, ref_source_list, train_target_list, ref_target_list = train_test_split(train_source_texts, train_target_texts, test_size=0.2,
                                                                          shuffle=False)

    all_texts = np.concatenate([train_source_texts.values, train_target_texts.values])
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token='<OOV>')
    tokenizer.fit_on_texts(all_texts)
    lens = len(tokenizer.word_index) + 1

    train_source_seqs = tokenizer.texts_to_sequences(train_source_list)
    ref_source_seqs = tokenizer.texts_to_sequences(ref_source_list)
    train_target_seqs = tokenizer.texts_to_sequences(train_target_list)
    test_source_seqs = tokenizer.texts_to_sequences(test_source_texts)

    max_sequence_length = max(len(seq) for seq in train_source_seqs + train_target_seqs + test_source_seqs)
    train_source_seqs = pad_sequences(train_source_seqs, maxlen=max_sequence_length, padding='post')
    train_target_seqs = pad_sequences(train_target_seqs, maxlen=max_sequence_length, padding='post')
    test_source_seqs = pad_sequences(test_source_seqs, maxlen=max_sequence_length, padding='post')
    ref_source_seqs = pad_sequences(ref_source_seqs, maxlen=max_sequence_length, padding='post')

    return max_sequence_length, train_source_seqs, train_target_seqs, test_source_seqs, lens, tokenizer, ref_source_seqs, ref_target_list





