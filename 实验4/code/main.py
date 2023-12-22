import pandas as pd
import numpy as np
from TextModel import TextModel
from data_processor import data_process
import keras
import os
from nltk.translate import bleu_score as bleu
import argparse
import tensorflow as tf
from keras.models import load_model


train_data = pd.read_csv("../data/train.csv")
test_data = pd.read_csv("../data/test.csv")

seed = 666
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default="train", help='test or train')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--model', default="GRU")

    args = parser.parse_args()

    max_sequence_length, train_source_seqs, train_target_seqs, test_source_seqs, lens, tokenizer, ref_source_seqs, ref_target_list = data_process(train_data,
                                                                                                          test_data)
    if args.run == "train":
        textModel = TextModel(max_sequence_length, lens)
        if args.model == "LSTM":
            model = textModel.build_LSTM_model()
        else:
            model = textModel.build_GRU_model()

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=["accuracy"])
        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath='TextModel.model',
            save_weights_only=False,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)

        history = model.fit([train_source_seqs, train_source_seqs], np.expand_dims(train_target_seqs, -1),
                            batch_size=64, epochs=args.epochs,
                            callbacks=[model_checkpoint_callback], validation_split=0.2, shuffle=False)
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        train_predictions = model.predict([ref_source_seqs, ref_source_seqs])
        train_predictions = np.argmax(train_predictions, axis=-1)
        train_predictions = tokenizer.sequences_to_texts(train_predictions.tolist())
        train_predictions = [text.replace('<OOV>', '') for text in train_predictions]
        train_predictions = pd.DataFrame(train_predictions, columns=['diagnosis'])

        re = ref_target_list.tolist()
        references = [[text.split()] for text in re]
        candidates = [text.split() for text in train_predictions['diagnosis']]

        bleu_score = bleu.corpus_bleu(references, candidates, weights=(0.25, 0.25, 0.25, 0.25))
        print("BLEU-4 评估指标：", bleu_score)

