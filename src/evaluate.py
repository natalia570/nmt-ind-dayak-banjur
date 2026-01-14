import numpy as np
import tensorflow as tf
import random

from nltk.translate.bleu_score import corpus_bleu
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

from src.dataset import load_dataset, prepare_dataset


# =====================================================
# Reproducibility
# =====================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# =====================================================
# Decode Function (Greedy)
# =====================================================
def decode_sequence(
    input_seq,
    encoder_model,
    decoder_model,
    tgt_tokenizer,
    tgt_max_len
):
    states_value = encoder_model.predict(input_seq, verbose=0)

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tgt_tokenizer.word_index["startseq"]

    decoded_sentence = []

    for _ in range(tgt_max_len):
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value,
            verbose=0
        )

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = tgt_tokenizer.index_word.get(sampled_token_index)

        if sampled_word == "endseq" or sampled_word is None:
            break

        decoded_sentence.append(sampled_word)

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]

    return decoded_sentence


# =====================================================
# Build Inference Models
# =====================================================
def build_inference_models(model):
    encoder_inputs = model.input[0]
    encoder_outputs, state_h_enc, state_c_enc = model.layers[4].output
    encoder_states = [state_h_enc, state_c_enc]

    encoder_model = tf.keras.Model(
        encoder_inputs,
        encoder_states
    )

    decoder_inputs = model.input[1]
    decoder_state_input_h = tf.keras.Input(shape=(state_h_enc.shape[-1],))
    decoder_state_input_c = tf.keras.Input(shape=(state_c_enc.shape[-1],))

    decoder_states_inputs = [
        decoder_state_input_h,
        decoder_state_input_c
    ]

    decoder_lstm = model.layers[6]
    decoder_outputs, state_h, state_c = decoder_lstm(
        model.layers[5].output,
        initial_state=decoder_states_inputs
    )

    decoder_dense = model.layers[7]
    decoder_outputs = decoder_dense(decoder_outputs)

    decoder_model = tf.keras.Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs, state_h, state_c]
    )

    return encoder_model, decoder_model


# =====================================================
# BLEU Evaluation
# =====================================================
def evaluate_bleu(
    model_path,
    test_csv,
    src_col,
    tgt_col,
    src_tokenizer,
    tgt_tokenizer,
    src_max_len,
    tgt_max_len
):
    model = load_model(model_path)

    encoder_model, decoder_model = build_inference_models(model)

    test_df = load_dataset(test_csv)
    test_df = prepare_dataset(
        test_df,
        src_col,
        tgt_col,
        add_tokens=False
    )

    references = []
    hypotheses = []

    for _, row in test_df.iterrows():
        input_seq = src_tokenizer.texts_to_sequences([row[src_col]])
        input_seq = pad_sequences(input_seq, maxlen=src_max_len, padding="post")

        decoded = decode_sequence(
            input_seq,
            encoder_model,
            decoder_model,
            tgt_tokenizer,
            tgt_max_len
        )

        references.append([row[tgt_col].split()])
        hypotheses.append(decoded)

    bleu = corpus_bleu(references, hypotheses)
    return bleu


# =====================================================
# Main
# =====================================================
if __name__ == "__main__":
    bleu_score = evaluate_bleu(
        model_path="models/nmt_model.h5",
        test_csv="data/sample/test_sample.csv",
        src_col="Bahasa Indonesia",
        tgt_col="Bahasa Banjur",
        src_tokenizer=None,  # load dari training
        tgt_tokenizer=None,  # load dari training
        src_max_len=30,
        tgt_max_len=30
    )

    print(f"BLEU Score: {bleu_score:.4f}")
