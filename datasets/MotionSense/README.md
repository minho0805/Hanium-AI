# 📱 MotionSense Dataset README

## 📌 Overview
MotionSense 데이터셋은 스마트폰(iPhone)의 센서를 활용하여 사람의 다양한 행동(Activity)을 기록한 **시계열(Time-series) 데이터셋**입니다.  

가속도계(Accelerometer), 자이로스코프(Gyroscope), 중력(Gravity), 자세(Attitude) 데이터를 포함하며, 인간 행동 인식(HAR: Human Activity Recognition)에 활용됩니다.


---

## 📁 폴더 설명

- `A_DeviceMotion_data/`  
  → 실제 센서 데이터

### 행동(Activity) 라벨

| 코드 | 의미 |
|------|------|
| dws | downstairs (계단 내려가기) |
| ups | upstairs (계단 올라가기) |
| wlk | walking |
| jog | jogging |
| sit | sitting |
| std | standing |

- `_1`, `_2` 등 숫자  
  → 동일 행동의 다른 실험 세션

---

## 👤 File Naming

```
sub_1.csv → 1번 사용자
sub_2.csv → 2번 사용자
```

👉 구조 요약  
```
행동 → 세션 → 사용자
```

---

## 📊 Data Format

각 CSV 파일은 다음 컬럼을 포함합니다:

```
attitude.roll
attitude.pitch
attitude.yaw

gravity.x
gravity.y
gravity.z

rotationRate.x
rotationRate.y
rotationRate.z

userAcceleration.x
userAcceleration.y
userAcceleration.z
```

---

## 🧠 Feature Description

| Feature | 설명 |
|--------|------|
| attitude | 기기의 방향 (Roll, Pitch, Yaw) |
| gravity | 중력 벡터 |
| rotationRate | 자이로 센서 값 |
| userAcceleration | 사용자 가속도 |

👉 핵심 Feature  
- `userAcceleration` → 움직임  
- `rotationRate` → 회전  

---

## ⚠️ Important Notes

- 본 데이터셋은 **모든 데이터가 정상 행동 데이터**
- 이상 행동(낙상 등)은 포함되어 있지 않음

---

## 🚀 How to Use

### 1. 데이터 로드

```python
import pandas as pd

df = pd.read_csv("A_DeviceMotion_data/wlk_7/sub_1.csv")
print(df.head())
```

---

### 2. 라벨 생성

```python
label = "walking"
df["label"] = label
```

---

### 3. 전체 데이터 통합

```python
import os
import pandas as pd

data = []
base_path = "A_DeviceMotion_data"

for activity in os.listdir(base_path):
    activity_path = os.path.join(base_path, activity)

    for file in os.listdir(activity_path):
        file_path = os.path.join(activity_path, file)

        df = pd.read_csv(file_path)
        df["label"] = activity.split("_")[0]
        data.append(df)

dataset = pd.concat(data, ignore_index=True)
```

---

### 4. 시계열 데이터 분할 (Sliding Window)

```python
import numpy as np

def create_sequences(data, window_size=100):
    sequences = []
    
    for i in range(len(data) - window_size):
        sequences.append(data[i:i+window_size])
        
    return np.array(sequences)
```

---

## 🔥 활용 방법

### ✔️ 1. 행동 분류 (Classification)
- walking vs jogging vs sitting 등

### ✔️ 2. 정상 데이터 기반 이상 탐지
- AutoEncoder 활용

```
정상 데이터로 학습 → reconstruction error로 이상 탐지
```

### ✔️ 3. 시계열 모델 적용
- LSTM
- GRU
- Transformer

---

## ❌ 한계

- 이상 데이터 없음
- 센서 위치 차이 존재 (예: 주머니 vs 손)

---

## 📌 추천 활용 전략

- 정상 데이터로만 학습 → 이상 탐지 모델 구축
- 일부 행동을 “가짜 이상”으로 라벨링하여 실험 가능

---

## 🏁 Summary

- 스마트폰 센서 기반 시계열 데이터셋
- 행동 인식(HAR)에 적합
- 이상 탐지에는 추가 전략 필요