from tensorflow import keras
import numpy as np
from tensorflow.keras.utils import to_categorical
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.optimizers import AdamW
import tensorflow as tf
from tensorflow.keras import models, layers
import tensorflow as tf
from tensorflow import keras
def twoTupleDic1():
    AA_list_sort = ['G', 'A', 'V', 'L', 'I', 'M', 'P', 'F', 'W', 'S', 'T', 'N', 'Q', 'Y', 'C', 'K', 'R', 'H', 'D', 'E', 'X','U','Z','B']
    AA_dict = {}
    numm = 1
    for i in AA_list_sort:
        AA_dict[i] = numm
        numm += 1
    return AA_dict

def ProSentence(pro, K):
	sentence = ""
	length = len(pro)
	for i in range(length - K + 1):
		sentence += pro[i: i + K] + " "
	return sentence
def load_dataset(path_data, filename, k=1):
    word_index = twoTupleDic1()
    df = pd.read_csv(path_data + filename, delimiter=',')
    texts = [ProSentence(seq, k) for seq in df['Sequence']]
    df['k_mer'] = texts
    sequences = []
    for sentence in texts:
        tokens = sentence.split()
        indices = [word_index[token] for token in tokens]
        sequences.append(indices)
    max_len = len(sequences[0])
    X = pad_sequences(sequences, maxlen=max_len)
    y = np.array(df['Label'])
    lb = LabelBinarizer()
    y = lb.fit_transform(y)
    y = to_categorical(y)
    return X, y


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import tensorflow as tf
from tensorflow.keras import layers, models


def create_model_cnn357_BiLSTM(vocab_size, max_sequence_length):
    # ------------------------------------------------------
    # Input Layer (Bắt buộc phải khai báo rõ ràng trong Functional API)
    # ------------------------------------------------------
    inputs = layers.Input(shape=(max_sequence_length,), name="input_layer")

    # ------------------------------------------------------
    # Embedding Layer
    # ------------------------------------------------------
    embedding = layers.Embedding(
        input_dim=vocab_size + 1,
        output_dim=300,
        trainable=True,
        name="embedding_layer"
    )(inputs)

    # ------------------------------------------------------
    # Rẽ nhánh Song song - 3 CNN Blocks với Kernel (3, 5, 7)
    # ------------------------------------------------------
    # Nhánh 1: Kernel Size = 3
    cnn_k3 = layers.Conv1D(filters=32, kernel_size=3, padding='same', name="conv_k3")(embedding)
    cnn_k3 = layers.BatchNormalization()(cnn_k3)
    cnn_k3 = layers.Activation('relu')(cnn_k3)
    cnn_k3 = layers.Dropout(0.2)(cnn_k3)

    # Nhánh 2: Kernel Size = 5
    cnn_k5 = layers.Conv1D(filters=32, kernel_size=5, padding='same', name="conv_k5")(embedding)
    cnn_k5 = layers.BatchNormalization()(cnn_k5)
    cnn_k5 = layers.Activation('relu')(cnn_k5)
    cnn_k5 = layers.Dropout(0.2)(cnn_k5)

    # Nhánh 3: Kernel Size = 7
    cnn_k7 = layers.Conv1D(filters=32, kernel_size=7, padding='same', name="conv_k7")(embedding)
    cnn_k7 = layers.BatchNormalization()(cnn_k7)
    cnn_k7 = layers.Activation('relu')(cnn_k7)
    cnn_k7 = layers.Dropout(0.2)(cnn_k7)

    # ------------------------------------------------------
    # Concatenate (Gộp đặc trưng theo trục channel)
    # ------------------------------------------------------
    # Shape sau gộp: (batch_size, max_sequence_length, 32 + 32 + 32 = 96)
    merged_features = layers.Concatenate(axis=-1, name="cnn_merge")([cnn_k3, cnn_k5, cnn_k7])

    # ------------------------------------------------------
    # BiLSTM Layer
    # ------------------------------------------------------
    bilstm = layers.Bidirectional(
        layers.LSTM(units=32, return_sequences=True),
        name="bilstm_layer"
    )(merged_features)

    # ------------------------------------------------------
    # Classifier (Phần phân loại)
    # ------------------------------------------------------
    flat = layers.Flatten()(bilstm)
    flat = layers.Dropout(0.4)(flat)

    dense1 = layers.Dense(64, activation='relu', name="dense_hidden")(flat)
    dense1 = layers.Dropout(0.3)(dense1)

    # CHÚ Ý: Đặt activation=None để trả về Logits thô phục vụ cho Distillation
    outputs = layers.Dense(2, activation=None, name="output_logits")(dense1)

    # Khởi tạo mô hình hoàn chỉnh bằng cách kết nối Đầu vào và Đầu ra
    model = models.Model(inputs=inputs, outputs=outputs, name="CNN3_Parallel_BiLSTM")

    return model

class Distiller(keras.Model):
    def __init__(self, student, teacher, temperature=3, alpha=0.7, ema=0.99):
        super(Distiller, self).__init__()
        self.student = student
        self.teacher = teacher
        self.temperature = temperature
        self.alpha = alpha
        self.ema = ema

        # SỬA LỖI 2: Đổi sang from_logits=True vì đầu ra mô hình hiện tại là Logits thô
        self.ce_loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)
        self.kld = keras.losses.KLDivergence()

    def compile(self, optimizer, metrics=None):
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)

    def train_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = None

        # Teacher dự đoán (Nhận Logits thô)
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Student dự đoán (Nhận Logits thô)
            student_predictions = self.student(x, training=True)

            # Hard Loss tính dựa trên Logits (Keras tự xử lý Softmax bên trong nhờ from_logits=True)
            student_loss = self.ce_loss_fn(y, student_predictions, sample_weight=sample_weight)

            # Soft Targets tính chuẩn xác từ Logits thô chia cho Temperature
            teacher_soft = tf.nn.softmax(teacher_predictions / self.temperature, axis=-1)
            student_soft = tf.nn.softmax(student_predictions / self.temperature, axis=-1)

            # KL Distillation Loss
            distill_loss = self.kld(teacher_soft, student_soft)
            distill_loss *= (self.temperature ** 2)

            # Total Loss
            total_loss = self.alpha * student_loss + (1.0 - self.alpha) * distill_loss

        # Cập nhật Student
        gradients = tape.gradient(total_loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.student.trainable_variables))

        # Cập nhật EMA cho Teacher
        for teacher_var, student_var in zip(self.teacher.variables, self.student.variables):
            teacher_var.assign(self.ema * teacher_var + (1.0 - self.ema) * student_var)

        return {
            "loss": total_loss,
            "student_loss": student_loss,
            "distill_loss": distill_loss
        }

    def test_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = None

        # Khi test, đầu ra dự đoán là Logits thô
        y_pred = self.student(x, training=False)
        student_loss = self.ce_loss_fn(y, y_pred, sample_weight=sample_weight)

        return {"student_loss": student_loss}
