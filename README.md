# 초기 충·방전 데이터 기반 ESS 배터리 수명 예측

**SKALA 3반 우강산 (U088)**

MIT–Stanford Battery Dataset · Regression · Batch 1 학습 / Batch 2 평가

## 프로젝트 목표

배터리의 정확한 수명을 알려면 셀이 수명을 다할 때까지 충·방전을 반복해야 한다.
이 프로젝트에서는 시험 시간을 줄이기 위해 **초기 100사이클 안에서 측정한 신호로
최종 Cycle Life를 예측할 수 있는지** 확인했다.

장·단수명 분류보다 사이클 단위 예측값이 셀 선별, 장기시험 우선순위, 교체 시점 검토에
더 유용하다고 판단해 회귀 문제를 선택했다.

| 구분 | 설정 |
|---|---|
| 데이터 | MIT–Stanford LFP/graphite cell 124개 |
| 개발 데이터 | Batch 1, 41 cells |
| 최종 평가 데이터 | Batch 2, 43 cells |
| Target | `cycle_life` |
| 입력 구간 | cycle 2–100 |
| 최종 모델 | Discharge features 10개 + Elastic Net |

---

## 1. 전략을 실제 Feature와 모델로 구현

### DAY 1 EDA에서 확인한 내용

#### Cycle Life 분포

Batch 1의 Cycle Life는 534–2237 cycles, 중앙값은 842 cycles다. Batch 2는
148–713 cycles, 중앙값은 481 cycles로 훨씬 짧고 범위도 좁다. Batch 2의 43개 셀 중
33개(76.7%)가 500 cycles 미만이고 1000 cycles를 넘는 셀은 없다.

두 Batch를 무작위로 섞으면 Batch 2의 특성이 학습 데이터에 들어가므로 성능이 실제보다
높게 보일 수 있다. 이 결과를 근거로 Batch 1에서만 모델을 개발하고 Batch 2는 마지막까지
분리했다. IQR 기준으로 드문 셀도 측정 오류라는 근거가 없으면 삭제하지 않았다.

#### 방전용량 열화곡선과 knee

초기 방전용량의 절대값은 셀 사이 차이가 작지만 cycle이 진행되면 감소 속도가 달라진다.
이를 반영해 cycle 2와 cycle 100의 방전용량, 2–100 구간 slope와 intercept를 피처로 만들었다.

전체 수명곡선에서 계산한 knee는 EDA 설명에는 도움이 되지만 모델 입력에는 넣지 않았다.
예측 시점 이후의 곡선을 사용해야 알 수 있는 값이므로 target leakage가 되기 때문이다.

#### ΔQ(V)

두 시점의 방전곡선을 동일한 전압 구간에서 비교했다.

```text
ΔQ(V) = Q(cycle 100, V) - Q(cycle 10, V)
```

Batch 1에서 `log_var_dq`와 Cycle Life의 Spearman 상관은 -0.882였다.
`log_min_dq`는 -0.854, `dq_mean`은 0.836이었다. 따라서 논문의 ΔQ 분산 피처를
기준선으로 두고 평균, 최솟값, 왜도, 첨도를 추가한 방전 피처 세트를 비교했다.

#### 충전 조건과 다중공선성

수명이 짧은 셀 중에는 비교적 이른 SOC에서 높은 두 번째 C-rate로 전환하는 셀이 있었다.
하지만 정책별 표본이 작고 정책 구성이 Batch와 함께 달라져 있어 인과관계로 해석하지 않았다.
`cc1`, `cc2`, `q1_pct`, 평균 충전시간은 건강 신호가 아니라 실험 처치를 나타낼 수 있어
최종 모델에서 제외하고 `policy-aware` 비교모델에서만 사용했다.

ΔQ(V) 통계량과 용량 추세 피처 사이에는 상관이 있고 학습 셀도 41개뿐이다. 일반
선형회귀보다 Ridge와 Elastic Net을 우선하고, Random Forest는 비선형 비교모델로 제한했다.

### EDA에서 구현으로 이어진 근거

| EDA 결과 | 구현 내용 | 판단 근거 |
|---|---|---|
| Batch별 수명 분포가 다름 | Batch 1 개발 → Batch 2 고정 평가 | 무작위 분할에 따른 과대평가 방지 |
| 수명 분포의 오른쪽 꼬리 | `log10(cycle_life)` 회귀 | 큰 target의 영향과 이분산 완화 |
| ΔQ(V)가 수명과 강한 단조관계 | 분산·최솟값·평균·왜도·첨도 | 초기 곡선 변화 반영 |
| 초기 용량은 비슷하지만 감소 속도는 다름 | Qd2, Qd100, slope, intercept | 초기 열화 속도 반영 |
| knee는 전체 수명이 지나야 확인 가능 | 입력 피처에서 제외 | target leakage 방지 |
| 피처 간 상관이 높고 표본이 작음 | Ridge·Elastic Net 우선 | 규제로 계수 변동 억제 |
| 충전정책과 Batch가 얽혀 있음 | policy-blind 본 모델 + policy-aware 비교 | 건강 신호와 처치 효과 구분 |

구현 위치: [`src/config.py`](src/config.py), [`src/features.py`](src/features.py),
[`src/models.py`](src/models.py)

---

## 2. Pipeline 개발

### 데이터 정리

원시 Batch 1에는 실험이 중간에 끊긴 셀이 있다. `b1c0`–`b1c4`는 Batch 2에서 시험이
이어졌으므로 기록을 연결해야 최종 수명이 맞는다. 저자 로더 기준에 따라 다섯 셀의 수명을
각각 1852, 2160, 2237, 1434, 1709 cycles로 반영했다.

이 처리를 생략하면 가장 오래 사용된 셀들의 target이 중간에서 잘린다. 모델 문제가 아니라
target 생성 오류로 결과가 달라지는 부분이므로 `prepare_data.py`와 테스트에서 따로 검증했다.

### 데이터 분할

```text
Batch 1, 41 cells
  ├─ 5-fold × 10-repeat CV: 모델 안정성 확인
  ├─ charge-policy Group CV: 보지 못한 정책에 대한 민감도 확인
  └─ 전체 41 cells: 최종 모델 재학습

Batch 2, 43 cells
  └─ 모델과 하이퍼파라미터 확정 후 1회 평가
```

Batch 3는 선택 과제에 해당한다. 현재 compact table의 셀 매핑이 저자 공식 제외 목록과
완전히 일치하지 않아 성능표에 넣지 않았다. 원본 `.mat`에서 다시 추출하고 cell ID를
대조한 뒤 평가하는 것이 맞다.

### 누수 방지

- 결측치 대치와 scaling은 sklearn Pipeline 안에서 각 CV 학습 fold에만 fit한다.
- `cycle_life`, `batch`, `partition`, `cell_id`는 입력 피처로 사용할 수 없도록 차단했다.
- 전체 수명곡선으로 계산되는 knee는 모델 피처에서 제외했다.
- 충전정책 변수는 본 모델 선택에서 제외하고 별도의 민감도 분석에만 사용했다.
- Batch 2 결과를 확인한 뒤 후보 모델이나 하이퍼파라미터를 변경하지 않았다.

### 핵심 피처

| 그룹 | 피처 | 의미 |
|---|---|---|
| ΔQ(V) | `log_var_dq`, `log_min_dq`, `dq_mean`, `dq_skew`, `dq_kurt` | 초기 방전곡선 형태의 변화 |
| Capacity | `cap_c2`, `cap_c100`, `cap_diff_100_2` | 초기 용량과 변화량 |
| Degradation trend | `cap_slope_2_100`, `cap_intercept_2_100` | 2–100사이클 열화 추세 |
| Sensor 후보 | 온도·내부저항 7개 | 방전 피처 대비 추가 정보 확인 |
| Policy ablation | `cc1`, `cc2`, `q1_pct`, `avg_charge_time` | 실험 처치 정보의 영향 확인 |

### 모델 선택 규칙

1. Batch 1 반복 CV MAPE가 최저값에서 1%p 이내인 후보를 남긴다.
2. 남은 후보 중 충전정책 Group CV MAPE가 가장 낮은 모델을 고른다.
3. 모델과 하이퍼파라미터를 확정한 뒤 Batch 2를 평가한다.

최종 Elastic Net의 `alpha`는 0.00464, `l1_ratio`는 0.7이다. 피처 수가 표본 수에 비해
많고 피처끼리 상관되어 있어 L1과 L2 규제를 함께 사용하는 구성이 적합하다고 판단했다.

### 전체 실행 흐름

```text
연장 셀 병합·데이터 계약 확인
             ↓
cycle 2–100 피처 준비
             ↓
Batch 1 모델·하이퍼파라미터 선택
             ↓
Batch 2 고정 평가
             ↓
성능표·셀별 예측·오류·분포 이동·예측구간 저장
```

```bash
git clone https://github.com/woorivermountain/skala-ess-battery-life-u088.git
cd skala-ess-battery-life-u088
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/prepare_data.py
python -m src.train --repeats 10 --bootstrap 5000
python -m src.visualize
pytest
```

구현 위치: [`src/preprocess.py`](src/preprocess.py), [`src/train.py`](src/train.py),
[`src/evaluate.py`](src/evaluate.py), [`scripts/run_pipeline.py`](scripts/run_pipeline.py),
[`tests/`](tests/)

---

## 3. 성능 리포팅과 목표 대비 GAP

### 후보 모델 비교

| 모델 | 피처 | Batch 1 반복 CV MAPE | 정책 Group CV | Batch 2 MAPE | 판단 |
|---|---|---:|---:|---:|---|
| Median baseline | 기준값 | 23.33% | 22.96% | 86.87% | 최소 기준 |
| Variance Ridge | ΔQ 분산 1개 | 9.17% | 8.92% | 32.43% | 논문 단일 피처 기준 |
| Discharge Ridge | 방전 10개 | 8.14% | 9.17% | 24.68% | 규제 선형 비교 |
| **Discharge Elastic Net** | **방전 10개** | **7.49%** | **8.21%** | **26.59%** | **최종 선택** |
| Random Forest | 방전 10개 | 10.27% | 11.82% | 40.79% | 외부평가에서 과적합 |
| Sensor Ridge | 방전+센서 17개 | 11.29% | 12.84% | 22.55% | 사후 비교만 수행 |
| Policy-aware Elastic Net | 센서+정책 21개 | 8.56% | 10.43% | 25.48% | 최종 선택에서 제외 |

Sensor Ridge는 Batch 2에서 22.55%로 최종 모델보다 낮은 MAPE를 냈다. 그러나 이 결과를
확인한 뒤 최종 모델을 바꾸면 Batch 2를 validation set으로 사용한 셈이 된다. 따라서
Batch 1에서 미리 정한 `discharge_elasticnet`을 최종 모델로 유지했다.

### 최종 모델 성능

| 평가 항목 | 결과 |
|---|---:|
| Batch 1 repeated CV MAPE | 7.49% ± 2.19% |
| Batch 1 policy Group CV MAPE | 8.21% |
| **Batch 2 MAPE** | **26.59%** |
| Batch 2 Median APE | 22.95% |
| Batch 2 MAE | 115.78 cycles |
| Batch 2 RMSE | 129.34 cycles |
| Batch 2 R² | -1.90 |
| MAPE bootstrap 95% CI | 21.19–34.02% |

### 원논문 대비 GAP

| 비교 기준 | 논문 수치 | 본 프로젝트 | GAP | 비교 조건 |
|---|---:|---:|---:|---|
| 과제 지정 headline test error | 9.1% | 26.59% | **+17.49%p** | 과제 요구 기준 |
| Table 1 discharge / primary / all cells | 13.0% | 26.59% | **+13.59%p** | 현재 모델과 가장 가까운 기준 |
| Table 1 full / primary / all cells | 14.1% | 26.59% | +12.49%p | 피처 구성이 다름 |
| Table 1 full / anomaly 제외 | 7.5% | 26.59% | 직접 비교하지 않음 | 이상 셀 제외 조건이 다름 |

9.1%, 13.0%, 7.5%는 같은 평가 조건에서 나온 수치가 아니다. 과제에서 요구한 9.1% GAP은
그대로 보고하되, 원인 분석에는 Batch 2 전체 셀을 평가한 13.0%를 함께 사용했다.

R²가 음수라는 점도 제외하지 않았다. Batch 1에서는 관계를 찾았지만 Batch 2의 평균 수명
수준이 크게 달라지면서 제곱오차 기준으로 단순 평균 예측보다 불리해졌다. 성능 저하의 핵심은
후보 모델 하나의 문제가 아니라 Batch 사이의 분포 이동에 있다.

근거 파일: [`results/model_selection.csv`](results/model_selection.csv),
[`results/model_performance.csv`](results/model_performance.csv),
[`results/uncertainty.json`](results/uncertainty.json)

---

## 4. 분석 결과의 도메인 해석과 한계

### 오류가 큰 셀

Batch 2의 실제 평균은 473.2 cycles였지만 예측 평균은 586.2 cycles였다. 전체적으로
약 113 cycles를 높게 예측했다.

| Cell | 실제 | 예측 | APE |
|---|---:|---:|---:|
| `b2c1` | 148 | 368 | 148.8% |
| `b2c20` | 502 | 791 | 57.6% |
| `b2c42` | 466 | 665 | 42.8% |

- 수명 500 미만 셀 MAPE: 28.15%
- 수명 500 이상 셀 MAPE: 21.41%
- `b2c1` 제외 민감도 MAPE: 23.68%

`b2c1`을 제외하면 수치는 좋아지지만 기본 성능표에서는 삭제하지 않았다. 원논문 평가에
포함된 셀이며, 명확한 측정 오류가 확인되지 않았기 때문이다.

### Batch 2에서 성능이 낮아진 이유

1. **Target shift**: 중앙값이 Batch 1의 842에서 Batch 2의 481 cycles로 이동했다.
2. **Feature shift**: `dq_mean`의 SMD는 -1.63, `cap_diff_100_2`는 -1.47,
   `cap_c100`은 -1.41이었다.
3. **짧은 수명 셀의 영향**: MAPE는 실제값이 작은 셀의 오차를 크게 반영한다.
4. **작은 표본**: 41개 셀에서 피처와 하이퍼파라미터를 고르므로 추정 변동이 크다.
5. **실험 조건의 결합**: 충전정책, 수집 시점, Batch가 독립적으로 변하지 않는다.

90% conformal prediction interval의 Batch 2 실제 coverage는 83.72%였다. Batch 1의
잔차로 만든 구간도 분포가 달라진 Batch 2에서는 그대로 보정되지 않았다.

### ESS 운영에서의 의미

현재 모델의 현실적인 용도는 정확한 교체일을 결정하는 것이 아니라, 초기 시험 결과로
추가 검사가 필요한 셀을 좁히는 **선별 도구**다. 다음과 같은 업무에 참고할 수 있다.

- 팩 조립 전 수명이 비슷한 셀의 1차 매칭
- 장기 열화시험 대상의 우선순위 결정
- 예상보다 빠르게 열화될 가능성이 있는 셀의 점검 순서 설정
- 충전정책별 잔차를 이용한 추가 실험 설계

현재 결과만으로 실제 BESS의 교체 또는 보증 의사결정을 자동화해서는 안 된다. 데이터가
한 종류의 LFP/graphite 셀과 실험실 급속충전 조건에 한정되어 있고, calendar aging,
주변 온도, SOC window, 제조 lot 차이가 포함되지 않았기 때문이다.

### 개선 우선순위

1. 원본 `.mat`에서 Batch 3를 다시 추출하고 공식 cell ID와 대조한다.
2. 다른 제조 lot와 운용 조건을 포함한 외부 검증을 수행한다.
3. leave-one-protocol-out CV로 새로운 충전정책에 대한 성능을 확인한다.
4. Batch별 calibration 또는 weighted conformal 방법을 비교한다.
5. 평균 예측값보다 예측구간을 활용하는 의사결정 기준을 설계한다.

세부 분석: [성능·오류 보고서](RESULTS.md) · [문헌 비교](research/literature_review.md) ·
[모델 카드](MODEL_CARD.md)

---

## 파일 구성

```text
├── data/
│   ├── README.md
│   ├── source/severson_features.csv
│   └── processed/severson_features.csv
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── preprocess.py
│   ├── features.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   └── visualize.py
├── results/
│   ├── model_selection.csv
│   ├── model_performance.csv
│   ├── batch2_predictions.csv
│   └── uncertainty.json
├── research/literature_review.md
├── MODEL_CARD.md
├── RUBRIC_CHECKLIST.md
├── requirements.txt
└── README.md
```

## 참고문헌

- Severson, K. A. et al. (2019). Data-driven prediction of battery cycle life before capacity degradation. *Nature Energy*, 4, 383–391. [Paper](https://doi.org/10.1038/s41560-019-0356-8) · [Code](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation)
- Geslin, A. et al. (2023). Selecting the appropriate features in battery lifetime predictions. *Joule*. [Paper](https://doi.org/10.1016/j.joule.2023.07.021) · [Code](https://github.com/geslina/Joule_2023_Perspective)
- Li, Y. et al. (2024). Predicting battery lifetime under varying usage conditions from early aging data. *Cell Reports Physical Science*. [Paper](https://doi.org/10.1016/j.xcrp.2024.101891)
- Attia, P. M. et al. Battery lifetime knee-point analysis and prediction. [Paper](https://www.sciencedirect.com/science/article/pii/S2666546820300069)

평가항목별 근거 파일은 [평가항목 확인표](RUBRIC_CHECKLIST.md)에 정리했다.
