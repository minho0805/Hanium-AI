# 📱 Smartphone Gait Dataset README

## 🧠 Overview

스마트폰 IMU 센서를 활용하여 수집된 인간 보행 시계열 데이터셋입니다.

본 프로젝트에서는 해당 데이터를 활용하여:

- 정상 보행 패턴 학습
- 보행 특징 추출
- TCN 기반 시계열 학습

을 수행합니다.

센서는 스마트폰 기반으로 수집되었으며,
허리(left waist) 또는 오른쪽 주머니(right pocket) 위치에서 기록되었습니다.

---

# ⚠️ Dataset Size Notice

GitHub 저장소 용량 제한으로 인해 본 프로젝트에는 일부 샘플 데이터(약 10명 사용자 데이터)만 포함되어 있습니다.

전체 Smartphone Gait Dataset은 아래 Kaggle 링크에서 다운로드할 수 있습니다.

https://www.kaggle.com/datasets/drdataboston/93-human-gait-database

전체 데이터셋에는 더 많은 사용자 및 보행 세션 데이터가 포함되어 있습니다.

---

# 📁 Dataset Structure

파일 구조:

```text
preprocessed_full/
└── subjects/
    ├── sub58-rp-s2.csv
    ├── sub91-rp-s2.csv
    ├── sub81-lw-s1.csv
    ├── ...
```

---

# 📌 File Naming Rule

```text
sub[번호]-[위치]-[세션].csv
```

예시:

```text
sub91-rp-s2.csv
```

의미:

| 값 | 설명 |
|---|---|
| sub91 | 91번 사용자 |
| rp | right pocket |
| lw | left waist |
| s1 / s2 | 실험 세션 |

---

# 📊 Data Characteristics

- 시계열(Time-Series) 데이터
- 스마트폰 IMU 센서 기반
- 정상 보행 데이터 중심
- 다수 사용자 포함
- 딥러닝(TCN) 학습 가능

---

# 📦 Sensor Features

데이터는 다음 센서 값을 포함합니다.

```text
accelerometer_x
accelerometer_y
accelerometer_z

gyroscope_x
gyroscope_y
gyroscope_z
```

※ 실제 컬럼명은 데이터셋 버전에 따라 다를 수 있습니다.

---

# ⚠️ Important Notes

## 1. 정상 보행 데이터

본 데이터셋은 정상 보행 데이터를 중심으로 구성되어 있습니다.

따라서:

- 정상 gait 학습
- gait feature extraction
- anomaly detection pretraining

등에 적합합니다.

---

## 2. 센서 위치 차이 존재

센서 위치:

- right pocket (rp)
- left waist (lw)

실제 사용자 환경에서는:

- 스마트폰 흔들림
- 방향 변화
- 주머니 마찰

등의 노이즈가 발생할 수 있습니다.

---

# 🚀 How to Use

## 데이터 로드

```python
import pandas as pd

df = pd.read_csv("subjects/sub91-rp-s2.csv")

print(df.head())
```

---

## Feature 선택

```python
features = [
    "accelerometer_x",
    "accelerometer_y",
    "accelerometer_z",
    "gyroscope_x",
    "gyroscope_y",
    "gyroscope_z"
]

X = df[features]
```

---

# 🔥 Usage in This Project

본 프로젝트에서는:

- Smartphone Gait Dataset → 정상 보행 데이터
- Daphnet Dataset → 이상 보행 데이터

를 결합하여

👉 정상 보행 vs 이상 보행 분류 모델을 구축합니다.

---

# 🧪 Model Pipeline

1. 데이터 전처리
2. Feature 추출
3. TCN 모델 학습
4. 이상 보행 탐지

---

# 🏁 Summary

- 스마트폰 IMU 기반 보행 데이터셋
- 정상 보행 학습에 적합
- TCN 기반 시계열 분석 가능
- 이상 탐지 모델의 사전 학습 데이터로 활용 가능