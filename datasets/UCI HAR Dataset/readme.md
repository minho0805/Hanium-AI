# UCI HAR Dataset

## 스마트폰을 이용한 인간 활동 인식 데이터셋

본 폴더는 **UCI Machine Learning Repository**에서 제공하는  
**Human Activity Recognition Using Smartphones Dataset**을 정리한 데이터셋 설명 문서입니다.

이 데이터셋은 허리에 관성 센서가 내장된 스마트폰을 착용한 사용자의 움직임을 측정하여,  
일상생활 활동을 분류하기 위해 구축된 인간 활동 인식 데이터셋입니다.

공식 링크:  
https://archive.ics.uci.edu/dataset/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones?utm_source=chatgpt.com

---

## 1. 데이터셋 개요

| 항목 | 내용 |
|---|---|
| 데이터셋 이름 | Human Activity Recognition Using Smartphones |
| 제공 기관 | UCI Machine Learning Repository |
| 공개일 | 2012년 12월 9일 |
| 데이터 유형 | 다변량, 시계열 데이터 |
| 주제 분야 | 컴퓨터 과학 |
| 관련 작업 | 분류, 클러스터링 |
| 인스턴스 수 | 10,299개 |
| 센서 | 스마트폰 내장 가속도계, 자이로스코프 |
| 측정 주파수 | 50Hz |
| 참가자 수 | 30명 |
| 참가자 연령 | 19세 ~ 48세 |

---

## 2. 데이터 수집 방식

본 데이터셋은 19세에서 48세 사이의 자원 참가자 30명을 대상으로 수집되었습니다.

각 참가자는 허리에 **Samsung Galaxy S II 스마트폰**을 착용한 상태에서 일상생활 활동을 수행했습니다.  
스마트폰에 내장된 가속도계와 자이로스코프를 이용하여 3축 선형 가속도와 3축 각속도 데이터를 50Hz의 일정한 속도로 측정했습니다.

실험 과정은 영상으로 녹화되었으며, 이후 수집된 센서 데이터에 대해 수동으로 활동 라벨이 부여되었습니다.

---

## 3. 활동 라벨

데이터셋에는 총 6개의 일상생활 활동 라벨이 포함되어 있습니다.

| 번호 | 활동 라벨 | 설명 |
|---|---|---|
| 1 | WALKING | 걷기 |
| 2 | WALKING_UPSTAIRS | 계단 오르기 |
| 3 | WALKING_DOWNSTAIRS | 계단 내려가기 |
| 4 | SITTING | 앉기 |
| 5 | STANDING | 서기 |
| 6 | LAYING | 눕기 |

본 프로젝트에서는 특히 다음 활동들이 스마트폰 기반 보행 분석과 관련성이 높습니다.

- WALKING
- WALKING_UPSTAIRS
- WALKING_DOWNSTAIRS

---

## 4. 데이터 전처리 정보

수집된 원본 센서 데이터는 고정 폭 슬라이딩 윈도우 방식으로 분할되었습니다.

| 항목 | 내용 |
|---|---|
| 윈도우 길이 | 2.56초 |
| 윈도우당 샘플 수 | 128개 |
| 중첩 비율 | 50% |
| Train/Test 분할 | 70% / 30% |
| 필터링 | Butterworth low-pass filter |
| 중력 성분 분리 기준 | 0.3Hz 저역 통과 필터 |

센서 신호는 신체 움직임에 의한 가속도 성분과 중력 성분으로 분리되었습니다.  
이후 시간 영역 및 주파수 영역의 다양한 feature가 계산되어 데이터셋에 포함되었습니다.

---

## 5. 데이터셋 폴더 구조

데이터셋을 다운로드하고 압축을 해제하면 일반적으로 다음과 같은 구조를 가집니다.

```text
UCI HAR Dataset/
├── README.txt
├── features.txt
├── features_info.txt
├── activity_labels.txt
├── train/
│   ├── X_train.txt
│   ├── y_train.txt
│   ├── subject_train.txt
│   └── Inertial Signals/
│       ├── body_acc_x_train.txt
│       ├── body_acc_y_train.txt
│       ├── body_acc_z_train.txt
│       ├── body_gyro_x_train.txt
│       ├── body_gyro_y_train.txt
│       ├── body_gyro_z_train.txt
│       ├── total_acc_x_train.txt
│       ├── total_acc_y_train.txt
│       └── total_acc_z_train.txt
└── test/
    ├── X_test.txt
    ├── y_test.txt
    ├── subject_test.txt
    └── Inertial Signals/
        ├── body_acc_x_test.txt
        ├── body_acc_y_test.txt
        ├── body_acc_z_test.txt
        ├── body_gyro_x_test.txt
        ├── body_gyro_y_test.txt
        ├── body_gyro_z_test.txt
        ├── total_acc_x_test.txt
        ├── total_acc_y_test.txt
        └── total_acc_z_test.txt
```

---

## 6. 주요 파일 설명

| 파일명 | 설명 |
|---|---|
| `activity_labels.txt` | 활동 번호와 활동 이름을 매핑한 파일 |
| `features.txt` | 추출된 feature 이름 목록 |
| `features_info.txt` | feature 계산 방식에 대한 설명 |
| `X_train.txt` | 학습용 feature 데이터 |
| `y_train.txt` | 학습용 활동 라벨 |
| `subject_train.txt` | 학습 데이터의 참가자 ID |
| `X_test.txt` | 테스트용 feature 데이터 |
| `y_test.txt` | 테스트용 활동 라벨 |
| `subject_test.txt` | 테스트 데이터의 참가자 ID |
| `Inertial Signals/` | 가속도계와 자이로스코프 기반 원시 시계열 신호 데이터 |

---

## 7. 본 프로젝트에서의 활용 목적

본 프로젝트는 스마트폰 IMU 센서를 기반으로 사용자의 보행 데이터를 분석하는 것을 목표로 합니다.

UCI HAR Dataset은 스마트폰에 내장된 가속도계와 자이로스코프를 사용하여 수집된 데이터이기 때문에,  
스마트폰 기반 보행 분석 모델을 실험하기 위한 기초 데이터셋으로 활용할 수 있습니다.

특히 다음과 같은 목적에 사용할 수 있습니다.

- 스마트폰 IMU 데이터 전처리 구조 이해
- 걷기, 계단 오르기, 계단 내려가기 활동 분류 실험
- LSTM, CNN, Random Forest 등 활동 인식 모델의 baseline 구축
- MotionSense, WISDM, HAPT 등 다른 스마트폰 센서 데이터셋과 비교 실험
- 향후 질환 의심 보행 분석 모델 개발을 위한 사전 실험 데이터로 활용

---

## 8. 한계점

이 데이터셋은 보행 분석 연구에 유용하지만, 다음과 같은 한계가 있습니다.

- 질병 환자 데이터가 아님
- 뇌졸중, 파킨슨병 등 질환 라벨이 포함되어 있지 않음
- 노인 대상 데이터가 아님
- 스마트폰 위치가 허리로 고정되어 있음
- 실제 일상 환경의 장시간 연속 측정 데이터는 아님

따라서 이 데이터셋은 질병을 직접 진단하기 위한 데이터셋이라기보다는,  
스마트폰 IMU 기반 인간 활동 인식 및 보행 분석 모델의 기초 실험용 데이터셋으로 사용하는 것이 적절합니다.

---

## 9. 데이터셋 사용 방법

데이터셋 원본 파일은 용량 문제로 GitHub 저장소에 직접 포함하지 않는 것을 권장합니다.

공식 UCI 링크에서 데이터셋을 다운로드한 뒤, 프로젝트 내 `datasets/` 폴더에 압축을 해제하여 사용합니다.

예시 구조는 다음과 같습니다.

```text
datasets/
└── UCI HAR Dataset/
    └── UCI HAR Dataset/
        ├── activity_labels.txt
        ├── features.txt
        ├── train/
        └── test/
```

---

## 10. GitHub 업로드 시 주의사항

데이터셋 파일은 용량이 크기 때문에 GitHub 저장소에 직접 업로드하지 않는 것이 좋습니다.

특히 다음과 같은 파일 또는 폴더는 `.gitignore`에 추가하는 것을 권장합니다.

```gitignore
# Dataset files
datasets/
data/
*.zip

# macOS metadata
__MACOSX/
.DS_Store
```

데이터셋은 GitHub에 직접 올리는 대신, README에 공식 다운로드 링크를 남기고  
팀원이 각자 동일한 공식 데이터셋을 내려받아 사용하는 방식이 가장 적절합니다.

---

## 11. 참고 링크

- UCI Machine Learning Repository  
  https://archive.ics.uci.edu/dataset/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones?utm_source=chatgpt.com

---

## 12. 참고 문헌

Reyes-Ortiz, J., Anguita, D., Ghio, A., Oneto, L., & Parra, X.  
Human Activity Recognition Using Smartphones Dataset.  
UCI Machine Learning Repository.
