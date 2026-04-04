import os
import glob
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# =========================
# 설정
# =========================
DATA_DIR = "/content/datasets/Daphnet"
WINDOW_SIZE = 64
STEP_SIZE = 32
TEST_SUBJECT = "S02"


# =========================
# 데이터 로드
# =========================
def load_file(file_path):
    df = pd.read_csv(file_path, sep=" ", header=None)
    df = df.iloc[:, :11]

    df.columns = [
        "time",
        "ankle_x", "ankle_y", "ankle_z",
        "thigh_x", "thigh_y", "thigh_z",
        "trunk_x", "trunk_y", "trunk_z",
        "label"
    ]

    file_name = os.path.basename(file_path).replace(".txt", "")
    df["subject"] = file_name[:3]
    df["session"] = file_name

    return df


def load_all():
    files = glob.glob(os.path.join(DATA_DIR, "*.txt"))
    dfs = [load_file(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


df = load_all()

print("전체 라벨 분포:")
print(df["label"].value_counts())


# =========================
# 전처리
# =========================
df = df[df["label"] != 0]

# 1 → 정상(0), 2 → FOG(1)
df["label"] = df["label"].apply(lambda x: 1 if x == 2 else 0)

print("\nBinary 라벨 분포:")
print(df["label"].value_counts())


feature_cols = [
    "ankle_x", "ankle_y", "ankle_z",
    "thigh_x", "thigh_y", "thigh_z",
    "trunk_x", "trunk_y", "trunk_z"
]


# =========================
# Sliding Window
# =========================
def create_windows(df):
    X, y, groups = [], [], []

    for session in df["session"].unique():
        temp = df[df["session"] == session]

        features = temp[feature_cols].values
        labels = temp["label"].values
        subject = temp["subject"].iloc[0]

        for i in range(0, len(temp) - WINDOW_SIZE + 1, STEP_SIZE):
            x = features[i:i+WINDOW_SIZE]
            y_window = labels[i:i+WINDOW_SIZE]

            label = Counter(y_window).most_common(1)[0][0]

            X.append(x)
            y.append(label)
            groups.append(subject)

    return np.array(X), np.array(y), np.array(groups)


X, y, groups = create_windows(df)

print("\nWindow shape:", X.shape)
print("라벨 분포:", Counter(y))


# =========================
# Train/Test (사람 기준)
# =========================
train_idx = groups != TEST_SUBJECT
test_idx = groups == TEST_SUBJECT

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print("\nTrain:", X_train.shape)
print("Test:", X_test.shape)


# =========================
# Scaling
# =========================
n_samples, time_steps, n_features = X_train.shape

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train.reshape(-1, n_features)).reshape(n_samples, time_steps, n_features)
X_test = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape[0], time_steps, n_features)


# =========================
# Class Weight
# =========================
weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weight = {0: weights[0], 1: weights[1]}
print("\nClass weight:", class_weight)


# =========================
# LSTM 모델
# =========================
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(time_steps, n_features)),
    Dropout(0.3),

    LSTM(32),
    Dropout(0.3),

    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Recall()]
)

model.summary()


# =========================
# 학습
# =========================
early_stop = EarlyStopping(patience=5, restore_best_weights=True)

model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=30,
    batch_size=64,
    class_weight=class_weight,
    callbacks=[early_stop]
)


# =========================
# 평가
# =========================
y_pred = (model.predict(X_test) > 0.5).astype(int)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nReport:")
print(classification_report(y_test, y_pred, target_names=["Normal", "FOG"]))


# =========================
# 저장
# =========================
model.save("daphnet_lstm.keras")