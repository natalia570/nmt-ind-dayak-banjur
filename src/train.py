import random
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Embedding, Dense,
    Bidirectional, Concatenate
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.dataset import load_dataset, prepare_dataset


# =====================================================
# Reproducibility
# =====================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# =====================================================
# Hyperparameters
# =====================================================
MAX_VOCAB_SIZE = 20000
EMBEDDING_DIM = 256
LATENT_DIM = 256
BATCH_SIZE = 64
EPOCHS = 20


# =====================================================
# Load & Prepare Data
# =====================================================
def load_training_data(train_path, val_path):
    train_df = load_dataset(train_path)
    val_df = load_dataset(val_path)

    train_df = prepare_dataset(
        train_df,
        src_col="Bahasa Indonesia",
        tgt_col="Bahasa Banjur",
        add_tokens=True
    )

    val_df = prepare_dataset(
        val_df,
        src_col="Bahasa Indonesia",
        tgt_col="Bahasa Banjur",
        add_tokens=True
    )

    return train_df, val_df


# =====================================================
# Tokenizer & Sequence Preparation
# =====================================================
def tokenize_and_pad(train_texts, val_texts, max_len=None):
    tokenizer = Tokenizer(
        num_words=MAX_VOCAB_SIZE,
        oov_token="<OOV>"
    )
    tokenizer.fit_on_texts(train_texts)

    train_seq = tokenizer.texts_to_sequences(train_texts)
    val_seq = tokenizer.texts_to_sequences(val_texts)

    if max_len is None:
        max_len = max(len(seq) for seq in train_seq)

    train_pad = pad_sequences(train_seq, maxlen=max_len, padding="post")
    val_pad = pad_sequences(val_seq, maxlen=max_len, padding="post")

    return tokenizer, train_pad, val_pad, max_len


# =====================================================
# Model Definition (Encoder–Decoder)
# =====================================================
def build_nmt_model(
    src_vocab_size,
    tgt_vocab_size,
    src_max_len,
    tgt_max_len
):
    encoder_inputs = Input(shape=(src_max_len,))
    enc_emb = Embedding(src_vocab_size, EMBEDDING_DIM)(encoder_inputs)
    encoder_outputs, state_h, state_c = LSTM(
        LATENT_DIM, return_state=True
    )(enc_emb)

    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(tgt_max_len,))
    dec_emb = Embedding(tgt_vocab_size, EMBEDDING_DIM)(decoder_inputs)
    decoder_lstm = LSTM(
        LATENT_DIM, return_sequences=True, return_state=True
    )
    decoder_outputs, _, _ = decoder_lstm(
        dec_emb, initial_state=encoder_states
    )

    decoder_dense = Dense(tgt_vocab_size, activation="softmax")
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model(
        [encoder_inputs, decoder_inputs],
        decoder_outputs
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# =====================================================
# Training Pipeline
# =====================================================
def main(train_csv, val_csv):
    train_df, val_df = load_training_data(train_csv, val_csv)

    src_tokenizer, X_train, X_val, src_max_len = tokenize_and_pad(
        train_df["Bahasa Indonesia"],
        val_df["Bahasa Indonesia"]
    )

    tgt_tokenizer, y_train, y_val, tgt_max_len = tokenize_and_pad(
        train_df["Bahasa Banjur"],
        val_df["Bahasa Banjur"]
    )

    y_train = y_train[:, 1:]
    y_val = y_val[:, 1:]

    model = build_nmt_model(
        src_vocab_size=len(src_tokenizer.word_index) + 1,
        tgt_vocab_size=len(tgt_tokenizer.word_index) + 1,
        src_max_len=src_max_len,
        tgt_max_len=tgt_max_len - 1
    )

    model.fit(
        [X_train, y_train[:, :-1]],
        y_train[..., None],
        validation_data=(
            [X_val, y_val[:, :-1]],
            y_val[..., None]
        ),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS
    )


if __name__ == "__main__":
    main(
        train_csv="data/train.csv",
        val_csv="data/val.csv"
    )
