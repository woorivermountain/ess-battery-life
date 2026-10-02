# 관련 연구 비교와 적용 범위

문헌을 인용하는 데 그치지 않고, 각 연구의 평가 조건이 이번 과제와 같은지 먼저 확인했다.
데이터셋, 피처, 이상치 처리, test partition이 다르면 MAPE 숫자만 직접 비교하기 어렵기 때문이다.

## 연구별 적용 내용

| 연구 | 데이터와 평가 | 이번 프로젝트에 반영한 내용 | 적용 시 주의점 |
|---|---|---|---|
| Severson et al. (2019) | LFP 124 cells, Batch 1/2/3 | ΔQ 분산 기준선, 로그 타깃, Elastic Net | 9.1%, Table 1 수치, 이상 셀 제외 수치의 조건이 다름 |
| Geslin et al. (2023) | 같은 공개 데이터의 피처 재검토 | 충전정책 피처를 별도 ablation으로 분리 | 충전 피처가 셀 상태보다 실험 처치를 인코딩할 수 있음 |
| Li et al. (2024) | NMC 225 cells, 여러 사용 조건 | in/out-of-distribution 성능을 구분 | 더 큰 NMC 결과를 41개 LFP 학습셋에 바로 적용할 수 없음 |
| Attia et al. | 수명곡선의 knee와 onset 분석 | knee를 EDA와 오류 분석에만 사용 | 전체 수명으로 구한 knee를 초기 예측 피처로 쓰면 누수 발생 |
| Zhang et al. (2025) | 다양한 열화 궤적의 표현학습 | 장기 개선 방향으로 검토 | n=41에서 deep learning을 쓰면 분산과 과적합 위험이 큼 |

## Severson 논문의 성능값을 구분한 이유

Severson 논문의 Table 1은 피처 모델과 test partition을 나눠 성능을 보고한다. Primary
test 전체 셀의 MAPE는 variance model 14.7%, discharge model 13.0%, full model
14.1%다. full model의 괄호 안 7.5%는 빠르게 실패한 셀 하나를 제외한 값이다.

과제에서 제시한 9.1%는 논문의 headline test error이므로 다음과 같이 구분했다.

| 용도 | 계산 |
|---|---:|
| 과제 지정 목표와 비교 | 26.59% - 9.1% = **+17.49%p** |
| 현재 피처 구성과 가까운 비교 | 26.59% - 13.0% = **+13.59%p** |
| 이상 셀 제외 수치 | 기본 성능과 직접 비교하지 않음 |

이번 결과에서도 `b2c1`의 APE가 148.8%로 가장 컸지만 기본 평가에서는 유지했다.
논문의 괄호 수치처럼 제외 조건을 바꿀 경우에는 민감도 분석이라고 따로 표시해야 한다.

## 충전 프로토콜을 별도로 다룬 이유

충전 프로토콜은 사용 목적에 따라 의미가 달라진다. 아직 운용 정책이 정해지지 않은 셀의
초기 상태를 비교한다면 충전정책은 실험 처치이므로 건강 피처와 분리하는 편이 맞다. 반대로
특정 운용 정책 아래의 수명을 예측한다면 알려진 공변량으로 사용할 수 있다.

이번 프로젝트는 셀의 초기 스크리닝을 주 용도로 정했다. 따라서 방전 진단 피처를 사용하는
모델을 본 모델로 두고, 충전정책을 포함한 모델은 별도 비교로만 남겼다. Policy-aware
Elastic Net은 Batch 1 CV 8.56%, Batch 2 MAPE 25.48%였다. 정책 정보를 추가해도
Batch 사이의 일반화 문제가 해결되지는 않았다.

## 결과를 해석할 때 넘지 말아야 할 범위

### 강한 상관은 인과관계가 아니다

ΔQ(V)와 Cycle Life의 상관이 높더라도 특정 열화 메커니즘의 원인이라고 단정할 수 없다.
본 프로젝트에서는 예측 피처로 사용하되 물리적 인과 증거로 표현하지 않았다.

### Batch 1 내부 성능이 현장 성능은 아니다

Batch 1 반복 CV MAPE는 7.49%였지만 Batch 2에서는 26.59%였다. 같은 Batch 안에서
무작위로 나눈 결과만으로 새로운 제조 lot나 운용 조건의 성능을 주장할 수 없다.

### 복잡한 모델이 항상 유리하지 않다

Random Forest의 Batch 1 적합 MAPE는 3.51%였지만 Batch 2 MAPE는 40.79%였다.
현재 표본 수에서는 모델 복잡도보다 규제와 외부 검증이 더 중요했다.

### MAPE 하나로는 오류 구조를 알 수 없다

MAPE는 실제 수명이 짧은 셀에 큰 비율오차를 부여한다. 따라서 MAE, RMSE, Median APE,
R²와 셀별 오류를 함께 보고했다.

### 예측구간도 외부 검증이 필요하다

nominal 90% conformal interval의 Batch 2 coverage는 83.72%였다. 학습 Batch의
잔차로 만든 구간도 분포 이동 아래에서는 과신될 수 있다.

### 내부 CV에도 선택 편향이 남을 수 있다

현재 반복 CV 결과는 같은 Batch 1에서 하이퍼파라미터를 고른 뒤 안정성을 확인한 값이다.
완전히 독립적인 nested CV 추정치는 아니므로 최종 일반화 성능으로 보지 않았다. 최종 성능은
모델 선택에 사용하지 않은 Batch 2 결과로 판단했다.

## 원문과 코드

- [Severson et al., Nature Energy 2019](https://web.mit.edu/braatzgroup/Severson_NatureEnergy_2019.pdf)
- [Severson official repository](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation)
- [Geslin et al., Joule 2023](https://www.cell.com/joule/fulltext/S2542-4351(23)00319-7)
- [Geslin feature-selection code](https://github.com/geslina/Joule_2023_Perspective)
- [Li et al., Cell Reports Physical Science 2024](https://www.cell.com/cell-reports-physical-science/fulltext/S2666-3864(24)00127-9)
- [Battery lifetime knee-point analysis](https://www.sciencedirect.com/science/article/pii/S2666546820300069)
- [Uncertainty-aware early prediction](https://pubs.rsc.org/en/content/articlelanding/2023/dd/d2dd00067a)
- [BatteryLife cross-dataset benchmark](https://arxiv.org/html/2502.18807v5)
- [Cross-diverse aging trajectories, Nature Machine Intelligence 2025](https://www.nature.com/articles/s42256-024-00972-x)
