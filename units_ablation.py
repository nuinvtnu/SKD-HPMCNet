def create_teacher(vocab_size, max_sequence_length):

    model = models.Sequential(name="Teacher")

    # Embedding
    model.add(
        layers.Embedding(
            input_dim=vocab_size + 1,
            output_dim=300,
            input_length=max_sequence_length,
            trainable=True
        )
    )

    # CNN Block 1
    model.add(
        layers.Conv1D(
            filters=64,
            kernel_size=3,
            padding='same'
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.2))

    # CNN Block 2
    model.add(
        layers.Conv1D(
            filters=64,
            kernel_size=3,
            padding='same'
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.2))

    # Bi-LSTM
    model.add(
        layers.Bidirectional(
            layers.LSTM(
                units=64,
                return_sequences=True
            )
        )
    )

    # Flatten
    model.add(layers.Flatten())

    # Dense
    model.add(layers.Dropout(0.5))
    model.add(
        layers.Dense(
            128,
            activation='relu'
        )
    )

    model.add(layers.Dropout(0.5))

    # Output
    model.add(
        layers.Dense(
            2,
            activation='softmax'
        )
    )

    return model

# ==========================================================
# Baseline Hybrid Model (No Knowledge Distillation)
# CNN(32)-CNN(32)-BiLSTM(32)
# ==========================================================
from tensorflow.keras import models, layers

def create_model_nokd(vocab_size, max_sequence_length):

    model = models.Sequential(name="Baseline_NoKD")

    # ------------------------------------------------------
    # Embedding
    # ------------------------------------------------------
    model.add(
        layers.Embedding(
            input_dim=vocab_size + 1,
            output_dim=300,
            input_length=max_sequence_length,
            trainable=True
        )
    )

    # ------------------------------------------------------
    # CNN Block 1
    # ------------------------------------------------------
    model.add(
        layers.Conv1D(
            filters=32,
            kernel_size=3,
            padding='same'
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.2))

    # ------------------------------------------------------
    # CNN Block 2
    # ------------------------------------------------------
    model.add(
        layers.Conv1D(
            filters=32,
            kernel_size=3,
            padding='same'
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.2))

    # ------------------------------------------------------
    # BiLSTM
    # ------------------------------------------------------
    model.add(
        layers.Bidirectional(
            layers.LSTM(
                units=32,
                return_sequences=True
            )
        )
    )

    # ------------------------------------------------------
    # Classifier
    # ------------------------------------------------------
    model.add(layers.Flatten())

    model.add(layers.Dropout(0.4))

    model.add(
        layers.Dense(
            64,
            activation='relu'
        )
    )

    model.add(layers.Dropout(0.3))

    model.add(
        layers.Dense(
            2,
            activation='softmax'
        )
    )

    return model
# ==========================================================
# Ablation Model: No CNN
# Embedding -> BiLSTM -> Dense
# ==========================================================
def create_model_nocnn(vocab_size, max_sequence_length):

    model = models.Sequential(name="NoCNN")

    # ------------------------------------------------------
    # Embedding Layer
    # ------------------------------------------------------
    model.add(
        layers.Embedding(
            input_dim=vocab_size + 1,
            output_dim=300,
            input_length=max_sequence_length,
            trainable=True
        )
    )

    # ------------------------------------------------------
    # BiLSTM
    # ------------------------------------------------------
    model.add(
        layers.Bidirectional(
            layers.LSTM(
                units=32,
                return_sequences=True
            )
        )
    )

    # ------------------------------------------------------
    # Classifier
    # ------------------------------------------------------
    model.add(layers.Flatten())

    model.add(layers.Dropout(0.4))

    model.add(
        layers.Dense(
            64,
            activation='relu'
        )
    )

    model.add(layers.Dropout(0.3))

    model.add(
        layers.Dense(
            2,
            activation='softmax'
        )
    )

    return model
# ==========================================================
# Ablation Model: LSTM
# Embedding -> CNN -> CNN -> LSTM -> Dense
# ==========================================================
def create_model_lstm(vocab_size, max_sequence_length):

    model = models.Sequential(name="CNN_LSTM")

    # ------------------------------------------------------
    # Embedding
    # ------------------------------------------------------
    model.add(
        layers.Embedding(
            input_dim=vocab_size + 1,
            output_dim=300,
            input_length=max_sequence_length,
            trainable=True
        )
    )

    # ------------------------------------------------------
    # CNN Block 1
    # ------------------------------------------------------
    model.add(
        layers.Conv1D(
            filters=32,
            kernel_size=3,
            padding='same'
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.2))

    # ------------------------------------------------------
    # CNN Block 2
    # ------------------------------------------------------
    model.add(
        layers.Conv1D(
            filters=32,
            kernel_size=3,
            padding='same'
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.2))

    # ------------------------------------------------------
    # LSTM (One-direction)
    # ------------------------------------------------------
    model.add(
        layers.LSTM(
            units=32,
            return_sequences=True
        )
    )

    # ------------------------------------------------------
    # Classifier
    # ------------------------------------------------------
    model.add(layers.Flatten())

    model.add(layers.Dropout(0.4))

    model.add(
        layers.Dense(
            64,
            activation='relu'
        )
    )

    model.add(layers.Dropout(0.3))

    model.add(
        layers.Dense(
            2,
            activation='softmax'
        )
    )

    return model
# Ablation Model: One CNN Layer
# Embedding -> CNN -> BiLSTM -> Dense
# ==========================================================

def create_model_1cnn(vocab_size, max_sequence_length):

    model = models.Sequential(name="CNN1_BiLSTM")

    # ------------------------------------------------------
    # Embedding
    # ------------------------------------------------------
    model.add(
        layers.Embedding(
            input_dim=vocab_size + 1,
            output_dim=300,
            input_length=max_sequence_length,
            trainable=True
        )
    )

    # ------------------------------------------------------
    # CNN Block (Only One)
    # ------------------------------------------------------
    model.add(
        layers.Conv1D(
            filters=32,
            kernel_size=3,
            padding='same'
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.2))

    # ------------------------------------------------------
    # BiLSTM
    # ------------------------------------------------------
    model.add(
        layers.Bidirectional(
            layers.LSTM(
                units=32,
                return_sequences=True
            )
        )
    )

    # ------------------------------------------------------
    # Classifier
    # ------------------------------------------------------
    model.add(layers.Flatten())

    model.add(layers.Dropout(0.4))

    model.add(
        layers.Dense(
            64,
            activation='relu'
        )
    )

    model.add(layers.Dropout(0.3))

    model.add(
        layers.Dense(
            2,
            activation='softmax'
        )
    )

    return model
# ==========================================================
# Ablation Model: Two CNN Layers
# Embedding -> CNN -> CNN -> CNN -> BiLSTM -> Dense
# ==========================================================
def create_model_2cnn(vocab_size, max_sequence_length):

    model = models.Sequential(name="CNN3_BiLSTM")

    # ------------------------------------------------------
    # Embedding
    # ------------------------------------------------------
    model.add(
        layers.Embedding(
            input_dim=vocab_size + 1,
            output_dim=300,
            input_length=max_sequence_length,
            trainable=True
        )
    )

    # ------------------------------------------------------
    # CNN Block 1
    # ------------------------------------------------------
    model.add(
        layers.Conv1D(
            filters=32,
            kernel_size=3,
            padding='same'
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.2))

    # ------------------------------------------------------
    # CNN Block 2
    # ------------------------------------------------------
    model.add(
        layers.Conv1D(
            filters=32,
            kernel_size=3,
            padding='same'
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.2))


    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.2))

    # ------------------------------------------------------
    # BiLSTM
    # ------------------------------------------------------
    model.add(
        layers.Bidirectional(
            layers.LSTM(
                units=32,
                return_sequences=True
            )
        )
    )

    # ------------------------------------------------------
    # Classifier
    # ------------------------------------------------------
    model.add(layers.Flatten())

    model.add(layers.Dropout(0.4))

    model.add(
        layers.Dense(
            64,
            activation='relu'
        )
    )

    model.add(layers.Dropout(0.3))

    model.add(
        layers.Dense(
            2,
            activation='softmax'
        )
    )

    return model
# ==========================================================
# Ablation Model: Three CNN Layers
# Embedding -> CNN -> CNN -> CNN -> BiLSTM -> Dense
# ==========================================================

def create_model_3cnn(vocab_size, max_sequence_length):

    model = models.Sequential(name="CNN3_BiLSTM")

    # ------------------------------------------------------
    # Embedding
    # ------------------------------------------------------
    model.add(
        layers.Embedding(
            input_dim=vocab_size + 1,
            output_dim=300,
            input_length=max_sequence_length,
            trainable=True
        )
    )

    # ------------------------------------------------------
    # CNN Block 1
    # ------------------------------------------------------
    model.add(
        layers.Conv1D(
            filters=32,
            kernel_size=3,
            padding='same'
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.2))

    # ------------------------------------------------------
    # CNN Block 2
    # ------------------------------------------------------
    model.add(
        layers.Conv1D(
            filters=32,
            kernel_size=3,
            padding='same'
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.2))

    # ------------------------------------------------------
    # CNN Block 3
    # ------------------------------------------------------
    model.add(
        layers.Conv1D(
            filters=32,
            kernel_size=3,
            padding='same'
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.2))

    # ------------------------------------------------------
    # BiLSTM
    # ------------------------------------------------------
    model.add(
        layers.Bidirectional(
            layers.LSTM(
                units=32,
                return_sequences=True
            )
        )
    )

    # ------------------------------------------------------
    # Classifier
    # ------------------------------------------------------
    model.add(layers.Flatten())

    model.add(layers.Dropout(0.4))

    model.add(
        layers.Dense(
            64,
            activation='relu'
        )
    )

    model.add(layers.Dropout(0.3))

    model.add(
        layers.Dense(
            2,
            activation='softmax'
        )
    )

    return model