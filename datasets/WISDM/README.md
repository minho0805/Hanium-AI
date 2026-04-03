# WISDM Dataset (IMU 기반 보행 데이터) 사용 가이드

## 📌 개요

WISDM (Wireless Sensor Data Mining) Dataset은 스마트폰의 가속도 센서를 활용하여 수집된 인간 활동 데이터셋으로, 보행 분석 및 시계열 기반 인공지능 모델 학습에 활용된다.

본 프로젝트에서는 해당 데이터셋을 활용하여 보행 패턴을 분석하고, LSTM 기반 시계열 모델 학습을 수행한다.

---

## 📊 데이터 구성

### 1. Raw Dataset (사용 대상)

데이터 파일: `WISDM_ar_v1.1_raw.txt`

#### ✔ 데이터 형식
```
[user],[activity],[timestamp],[x],[y],[z];
```
#### ✔ 예시
```
33,Jogging,49105962326000,-0.6946377,12.680544,0.50395286;
```

#### ✔ 컬럼 설명

| 컬럼 | 설명 |
|------|------|
| user | 사용자 ID (1~36) |
| activity | 수행 행동 (Walking, Jogging 등) |
| timestamp | 시간 정보 (nanoseconds) |
| x | X축 가속도 |
| y | Y축 가속도 |
| z | Z축 가속도 |

#### ✔ 주요 특징

- 샘플링 주파수: 약 20Hz
- 총 데이터 수: 약 1,000,000+ samples
- 센서: 스마트폰 가속도계 (Accelerometer)

---


## 🎯 사용 목적

본 데이터셋은 다음과 같은 목적에 활용된다.

- 보행 패턴 분석
- 시계열 데이터 기반 행동 분류
- LSTM 모델 학습 및 성능 검증

---

## ⚙️ 데이터 전처리 방법

### 1. 데이터 로드

- CSV 형태로 파싱
- ";" 제거 후 데이터 분리

---

### 2. Sliding Window 적용

시계열 데이터를 일정 길이로 분할하여 모델 입력으로 사용

python
window_size = 40  # 약 2초 (20Hz 기준)

### 3. 입력 데이터 형태

(batch_size, time_steps, features)

예시
(128, 40, 3)

	•	features: x, y, z

⸻

4. 라벨 구성
	•	Walking → 정상 보행
	•	기타 활동 → 비정상(유사 패턴)

⸻

⚠️ 한계점 및 보완 전략

1. Gyroscope 데이터 없음
	•	가속도 데이터만 존재
	•	추후 직접 수집 데이터로 보완 필요

⸻

2. 질병 데이터 없음
	•	실제 이상 보행(파킨슨 등) 데이터 포함 X
	•	Daphnet Dataset과 결합 필요

⸻

3. 샘플링 주파수 차이
	•	WISDM: 20Hz
	•	실제 목표: 50Hz

→ 실제 스마트폰 데이터 수집 후 Fine-tuning 수행

⸻

🚀 활용 전략
	1.	WISDM 데이터로 기본 LSTM 모델 학습
	2.	Daphnet 데이터로 이상 보행 학습 추가
	3.	직접 수집 데이터로 모델 성능 개선 (Fine-tuning)

