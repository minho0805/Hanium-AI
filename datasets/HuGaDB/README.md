## 📊 HuGaDB - Human Gait Database

### 🧠 Overview
이 데이터셋은 **건강한 성인의 일상 보행 활동**을 수집한 데이터로,  
6개의 IMU 센서(허벅지, 정강이, 발)를 통해 **걷기, 달리기, 계단, 앉기** 등의  
다양한 활동을 연속적으로 기록한 시계열 데이터입니다.

본 프로젝트에서는 **정상 보행 패턴 학습**을 위한 기준 데이터로 활용하며,  
특히 **허벅지 IMU 센서 데이터**를 중심으로 보행 특징을 추출합니다.

---

### 📁 Dataset Structure
HuGaDB/
├── User_1/
│   ├── User1_1_1.txt  # 사용자 1, 세션 1, 기록 1
│   ├── User1_1_2.txt  # 사용자 1, 세션 1, 기록 2
│   ├── User1_2_1.txt  # 사용자 1, 세션 2, 기록 1
│   └── ...
├── User_2/
│   ├── User2_1_1.txt
│   └── ...
└── User_18/
└── ...
**파일명 규칙:**  
`UserX_Y_Z.txt`
- X: 사용자 번호 (1~18)
- Y: 세션 번호 (1~2)
- Z: 기록 번호 (1~n)

각 파일은 **연속된 활동 조합**을 포함합니다.  
예: 앉기 → 일어서기 → 걷기 → 계단 오르기 → 걷기 → 앉기

---

### 🧾 Columns Description

**총 39개 컬럼 (헤더 없음)**

| Index | Sensor | Description |
|-------|--------|-------------|
| 0-5 | **Right Foot** | AccX, AccY, AccZ, GyrX, GyrY, GyrZ |
| 6-11 | **Right Shin** | AccX, AccY, AccZ, GyrX, GyrY, GyrZ |
| 12-17 | **Right Thigh** ⭐ | AccX, AccY, AccZ, GyrX, GyrY, GyrZ |
| 18-23 | **Left Foot** | AccX, AccY, AccZ, GyrX, GyrY, GyrZ |
| 24-29 | **Left Shin** | AccX, AccY, AccZ, GyrX, GyrY, GyrZ |
| 30-35 | **Left Thigh** ⭐ | AccX, AccY, AccZ, GyrX, GyrY, GyrZ |
| 36-37 | **EMG** | Right EMG, Left EMG |
| 38 | **Label** | Activity ID (1~11) |

**센서 위치:**
허벅지 (Thigh) ← 프로젝트 핵심
         ↓
    정강이 (Shin)
         ↓
      발 (Foot)

---

### 🏷️ Label Definition

| ID | Activity | Korean | 사용 여부 |
|----|----------|--------|----------|
| 1 | **Walking** | 걷기 | ✅ 사용 |
| 2 | Running | 달리기 | ✅ 사용 |
| 3 | Going up | 계단 오르기 | ✅ 사용 |
| 4 | Going down | 계단 내려가기 | ✅ 사용 |
| 5 | Sitting | 앉아있기 | ⚪ 선택 |
| 6 | Standing | 서있기 | ⚪ 선택 |
| 7 | Biking | 자전거 | ❌ 제외 |
| 8 | Sitting down | 앉는 동작 | ❌ 제외 |
| 9 | Standing up | 일어서는 동작 | ❌ 제외 |
| 10 | Shift to right | 우측 이동 | ❌ 제외 |
| 11 | Shift to left | 좌측 이동 | ❌ 제외 |

---

### ⚙️ Preprocessing

#### 1. 데이터 로드 및 컬럼명 지정

```python
import pandas as pd
import numpy as np

# 컬럼명 정의
columns = [
    'RF_AccX', 'RF_AccY', 'RF_AccZ', 'RF_GyrX', 'RF_GyrY', 'RF_GyrZ',
    'RS_AccX', 'RS_AccY', 'RS_AccZ', 'RS_GyrX', 'RS_GyrY', 'RS_GyrZ',
    'RT_AccX', 'RT_AccY', 'RT_AccZ', 'RT_GyrX', 'RT_GyrY', 'RT_GyrZ',
    'LF_AccX', 'LF_AccY', 'LF_AccZ', 'LF_GyrX', 'LF_GyrY', 'LF_GyrZ',
    'LS_AccX', 'LS_AccY', 'LS_AccZ', 'LS_GyrX', 'LS_GyrY', 'LS_GyrZ',
    'LT_AccX', 'LT_AccY', 'LT_AccZ', 'LT_GyrX', 'LT_GyrY', 'LT_GyrZ',
    'R_EMG', 'L_EMG', 'Activity'
]

# 데이터 로드
df = pd.read_csv('User_1/User1_1_1.txt', header=None, names=columns)
```

#### 2. 허벅지 센서만 추출 (프로젝트 핵심)

```python
# 오른쪽 + 왼쪽 허벅지 데이터
thigh_columns = [
    'RT_AccX', 'RT_AccY', 'RT_AccZ', 'RT_GyrX', 'RT_GyrY', 'RT_GyrZ',
    'LT_AccX', 'LT_AccY', 'LT_AccZ', 'LT_GyrX', 'LT_GyrY', 'LT_GyrZ',
    'Activity'
]

df_thigh = df[thigh_columns]
```

#### 3. 보행 관련 활동만 필터링

```python
# 걷기(1), 달리기(2), 계단 오르기(3), 계단 내려가기(4)만 사용
walking_activities = [1, 2, 3, 4]
df_walking = df_thigh[df_thigh['Activity'].isin(walking_activities)]
```

#### 4. 정상 보행 라벨로 통일

```python
# 모두 정상 보행(0)으로 라벨링
df_walking['label'] = 0  # 0 = Normal gait
```

---

### 📊 Data Characteristics

#### 1. 시계열 데이터
- **샘플링 레이트**: 60 Hz
- **연속 기록**: 활동 전환 포함
- **총 길이**: 약 10시간 (18명 합산)

#### 2. 센서 특성
가속도계 (Accelerometer)

범위: ±16g
단위: m/s² 또는 g
축: X (좌우), Y (전후), Z (상하)

자이로스코프 (Gyroscope)

범위: ±2000°/s
단위: deg/s
축: X (Roll), Y (Pitch), Z (Yaw)

#### 3. 데이터 분포
- **Walking (걷기)**: ~679,000 샘플 (가장 많음) ⭐
- **Running (달리기)**: ~187,000 샘플
- **계단 오르기/내리기**: 각 ~28,000 샘플

---

### ⚠️ Important Considerations

#### 1. 연속 활동 전환
활동이 연속적으로 기록되므로 **전환 구간 처리 필요**

```python
# 활동 변화 지점 탐지
activity_changes = df['Activity'].diff() != 0

# 전환 구간 제거 (앞뒤 1초씩)
transition_window = 60  # 60 Hz × 1초
mask = activity_changes.rolling(window=transition_window*2, center=True).sum() == 0
df_clean = df[mask]
```

#### 2. 센서 방향 정규화
허벅지 센서는 **부착 방향이 일정하지 않을 수 있음**

```python
# 중력 방향 정규화 (Z축 평균이 9.8m/s²에 가깝게)
gravity = 9.81
df['RT_AccZ_norm'] = df['RT_AccZ'] - df['RT_AccZ'].rolling(100).mean()
```

#### 3. Subject-dependent Split
사용자별로 보행 패턴이 다르므로  
**train/test를 사용자 기준으로 분리**

```python
# User 1~14: Train
# User 15~18: Test
train_users = [f'User_{i}' for i in range(1, 15)]
test_users = [f'User_{i}' for i in range(15, 19)]
```

---

### 🚀 Usage in This Project

본 프로젝트에서는:
- **HuGaDB → 정상 보행 데이터** ✅
- **Daphnet → 이상 보행 데이터 (FOG)** ✅

를 결합하여  
👉 **정상 vs 이상 보행 분류 모델**을 구축합니다.

#### 역할 분담
HuGaDB (허벅지 센서)
└─ 정상 보행 패턴 학습
├─ Walking (걷기)
├─ Running (달리기)
└─ Stairs (계단)
Daphnet (허벅지 센서)
└─ 이상 보행 패턴 학습
└─ Freezing of Gait (FOG)

---

### 🧪 Model Pipeline

#### Step 1: 데이터 전처리
```python
# 1. 허벅지 센서 추출
# 2. 보행 활동 필터링 (1, 2, 3, 4)
# 3. 정상 라벨(0) 할당
# 4. 전환 구간 제거
```

#### Step 2: Sliding Window 적용
```python
def create_windows(data, window_size=120, stride=60):
    """
    window_size: 120 samples = 2초 (60Hz)
    stride: 60 samples = 1초 overlap
    """
    windows = []
    labels = []
    
    for i in range(0, len(data) - window_size, stride):
        window = data.iloc[i:i+window_size, :-1].values
        label = data.iloc[i:i+window_size, -1].mode()[0]
        
        windows.append(window)
        labels.append(label)
    
    return np.array(windows), np.array(labels)

X, y = create_windows(df_walking, window_size=120, stride=60)
print(f"X shape: {X.shape}")  # (num_windows, 120, 12)
print(f"y shape: {y.shape}")  # (num_windows,)
```

#### Step 3: 특징 정규화
```python
from sklearn.preprocessing import StandardScaler

# 각 센서 채널별로 정규화
scaler = StandardScaler()
X_reshaped = X.reshape(-1, X.shape[-1])
X_scaled = scaler.fit_transform(X_reshaped)
X_scaled = X_scaled.reshape(X.shape)
```

#### Step 4: LSTM 모델 학습
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential([
    LSTM(64, input_shape=(120, 12), return_sequences=True),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')  # 0: Normal, 1: Abnormal
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
```

#### Step 5: 모델 평가
```python
from sklearn.metrics import classification_report, confusion_matrix

y_pred = (model.predict(X_test) > 0.5).astype(int)

print(classification_report(y_test, y_pred, 
                          target_names=['Normal', 'Abnormal']))
```

---

### 💡 Expected Results

#### 정상 보행 탐지 성능
HuGaDB 정상 보행 특징:
✅ 규칙적인 보행 주기
✅ 좌우 대칭적 움직임
✅ 일정한 보폭 간격
✅ 안정적인 속도 유지

#### Daphnet 이상 보행과의 차이점
FOG (Freezing of Gait) 특징:
❌ 갑작스러운 보행 중단
❌ 불규칙한 진동 패턴
❌ 좌우 비대칭성 증가
❌ 보폭 감소 또는 멈춤
---

### 📦 Download & Quick Start

```bash
# Kaggle API로 다운로드
kaggle datasets download -d romanchereshnev/hugadb-human-gait-database
unzip hugadb-human-gait-database.zip

# 또는 GitHub에서 직접 다운로드
git clone https://github.com/romanchereshnev/HuGaDB.git
```

```python
# Quick Start
import pandas as pd

# 데이터 로드
df = pd.read_csv('HuGaDB/User_1/User1_1_1.txt', header=None)

# 허벅지 데이터 확인
thigh_data = df.iloc[:, 12:36]  # 인덱스 12-35: 양쪽 허벅지
print(f"Data shape: {thigh_data.shape}")
print(f"Walking samples: {(df.iloc[:, -1] == 1).sum()}")
```

---

### 📚 References

- **Paper**: Chereshnev & Kertész-Farkas (2018), "HuGaDB: Human Gait Database for Activity Recognition from Wearable Inertial Sensor Networks"
- **Kaggle**: https://www.kaggle.com/datasets/romanchereshnev/hugadb-human-gait-database
- **GitHub**: https://github.com/romanchereshnev/HuGaDB