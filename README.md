# 초기 충·방전 데이터로 LFP 셀 수명 예측

**초기 100사이클의 신호로 최종 배터리 수명을 예측하고, 다른 Batch에서의 성능을 확인한 프로젝트입니다.** ESS 셀 선별을 염두에 두고 MIT–Stanford의 LFP/graphite 실험실 데이터를 사용했습니다. Batch 1에서 선택한 모델을 Batch 2에 평가하고, 개발 성능과 외부 평가의 차이를 함께 기록했습니다.

[성능·오류 보고서](RESULTS.md) · [모델 카드](MODEL_CARD.md) · [문헌 비교](research/literature_review.md) · [평가항목 확인표](RUBRIC_CHECKLIST.md)

`Python` · `scikit-learn` · `Leakage-safe Pipeline` · `Group CV` · `Distribution Shift`

SKALA 3반 우강산의 회귀 모델링 프로젝트입니다.

## 문제를 좁히고 구현한 과정

배터리 수명을 직접 확인하려면 셀이 수명을 다할 때까지 충·방전을 반복해야 합니다. 초기 시험만으로 추가 검사가 필요한 셀을 가려낼 수 있는지 알아보고, 장·단수명 분류 대신 사이클 단위의 `cycle_life` 회귀를 선택했습니다.

학습 셀이 41개이고 피처 사이의 상관이 높아 Ridge와 Elastic Net을 우선 검토했습니다. 각 CV fold 안에서 전처리를 학습하고, 예측 시점 이후에야 알 수 있는 knee는 입력에서 제외했습니다. Batch 1에서 최종 모델을 선택한 뒤 Batch 2의 오차와 분포 차이를 확인했습니다.

## 담당 역할과 대표 결과

데이터 정리, EDA, 초기 cycle 피처, 전처리·모델 선택 파이프라인, 외부 Batch 평가와 오류 분석을 구현했습니다. 논문 수치와 과제 기준을 같은 조건의 결과로 섞지 않고, 비교 조건과 차이를 보고서에 연결했습니다.

| 항목 | 설정·결과 |
| --- | --- |
| 데이터셋 | MIT–Stanford LFP/graphite **124개 셀** 중 Batch 1 **41개**, Batch 2 **43개**를 사용했습니다. |
| 입력·최종 모델 | cycle **2–100**, 방전 피처 **10개**와 Discharge Elastic Net입니다. |
| Batch 1 반복 CV | MAPE **7.49% ± 2.19%**, 정책 Group CV **8.21%**였습니다. |
| Batch 2 고정 평가 | MAPE **26.59%**, R² **-1.90**였습니다. |
| 사후 후보 비교 | Sensor Ridge의 Batch 2 MAPE는 **22.55%**였지만 최종 모델을 바꾸지 않았습니다. |

최종 모델은 Batch 1에서 정한 `discharge_elasticnet`으로 유지했습니다. Batch 2 결과로 모델을 다시 고르는 대신, 타깃·입력 분포 차이와 큰 오차가 난 셀을 다음 검증 과제로 남겼습니다.

## 빠른 시작

```bash
git clone https://github.com/woorivermountain/ess-battery-life.git
cd ess-battery-life
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/prepare_data.py
python -m src.train --repeats 10 --bootstrap 5000
python -m src.visualize
pytest
```

## 다음으로 볼 문서

- [성능·오류 보고서](RESULTS.md)는 셀별 예측과 외부 평가 차이를 설명합니다.
- [문헌 비교](research/literature_review.md)는 원논문의 피처·평가 조건을 정리합니다.
- [모델 카드](MODEL_CARD.md)는 입력·출력, 사용 범위와 모니터링 항목을 설명합니다.
- [평가항목 확인표](RUBRIC_CHECKLIST.md)는 과제 기준별 근거 파일을 연결합니다.
- 아래 접힌 절에 EDA, 파이프라인, 전체 성능표와 참고문헌을 보존했습니다.

<details>
<summary>EDA에서 피처와 모델로 이어진 선택</summary>

### 수명 분포

Batch 1의 Cycle Life는 **534–2237 cycles**, 중앙값은 **842 cycles**였습니다. Batch 2는 **148–713 cycles**, 중앙값은 **481 cycles**였습니다. Batch 2의 43개 셀 중 **33개(76.7%)**가 500 cycles 미만이며, 1000 cycles를 넘는 셀은 없었습니다.

두 Batch를 무작위로 섞지 않고 Batch 1에서 모델을 개발했습니다. Batch 2는 모델 선택 점수로 사용하지 않았습니다. IQR 기준으로 드문 셀도 측정 오류라는 근거가 없으면 삭제하지 않았습니다.

### 방전용량과 knee

초기 방전용량의 절대값은 셀 사이 차이가 작지만 cycle이 진행되면 감소 속도가 달랐습니다. cycle 2와 cycle 100의 방전용량, 2–100 구간의 slope·intercept를 피처로 만들었습니다.

전체 수명곡선으로 계산한 knee는 EDA 설명에 사용하고 입력에서는 제외했습니다. 예측 시점 이후의 곡선을 사용하므로 target leakage가 생길 수 있기 때문입니다.

### ΔQ(V)

동일한 전압 구간에서 두 시점의 방전곡선을 비교했습니다.

```text
ΔQ(V) = Q(cycle 100, V) - Q(cycle 10, V)
```

Batch 1에서 Cycle Life와의 Spearman 상관은 `log_var_dq` **-0.882**, `log_min_dq` **-0.854**, `dq_mean` **0.836**이었습니다. 논문의 ΔQ 분산 피처를 기준선으로 두고 평균·최솟값·왜도·첨도를 추가한 피처 세트를 비교했습니다.

### 충전 조건과 규제 모델

수명이 짧은 셀 중에는 비교적 이른 SOC에서 높은 두 번째 C-rate로 전환하는 셀이 있었습니다. 정책별 표본이 작고 Batch와 정책 구성이 함께 달라져, 이 관찰을 인과관계로 해석하지 않았습니다. `cc1`, `cc2`, `q1_pct`, 평균 충전시간은 실험 처치를 나타낼 수 있어 본 모델에서 제외하고 `policy-aware` 비교에만 사용했습니다.

ΔQ(V) 통계량과 용량 추세 피처가 상관되고 학습 셀이 41개라, 규제 선형 모델인 Ridge·Elastic Net을 우선했습니다. Random Forest는 비선형 비교모델로 두었습니다.

| EDA 결과 | 구현 내용 | 선택 이유 |
| --- | --- | --- |
| Batch별 수명 분포가 달랐습니다. | Batch 1 개발 → Batch 2 고정 평가로 구성했습니다. | 개발 Batch 밖의 성능을 확인합니다. |
| 수명 분포에 오른쪽 꼬리가 있었습니다. | `log10(cycle_life)`를 회귀합니다. | 큰 타깃의 영향과 이분산을 완화합니다. |
| ΔQ(V)와 수명에 강한 단조관계가 있었습니다. | 분산·최솟값·평균·왜도·첨도를 사용합니다. | 초기 곡선 변화를 반영합니다. |
| 초기 용량보다 감소 속도가 달랐습니다. | Qd2, Qd100, slope, intercept를 사용합니다. | 초기 열화 추세를 반영합니다. |
| knee는 전체 수명이 지나야 계산할 수 있습니다. | 입력 피처에서 제외했습니다. | 예측 시점 이후의 정보 사용을 막습니다. |
| 피처가 상관되고 표본이 작았습니다. | Ridge·Elastic Net을 우선했습니다. | 규제로 계수 변동을 억제합니다. |
| 충전정책과 Batch가 얽혀 있었습니다. | policy-blind 본 모델과 policy-aware 비교를 분리했습니다. | 건강 신호와 처치 정보를 구분합니다. |

구현 위치: [`src/config.py`](src/config.py), [`src/features.py`](src/features.py), [`src/models.py`](src/models.py)

</details>

<details>
<summary>데이터 정리·누수 방지·모델 선택 파이프라인</summary>

### 연장 셀의 기록 연결

원시 Batch 1의 `b1c0`–`b1c4`는 Batch 2에서 시험이 이어졌습니다. 저자 로더 기준으로 다섯 셀의 최종 수명을 **1852, 2160, 2237, 1434, 1709 cycles**로 반영했습니다. 중간에 끊긴 기록을 최종 수명으로 쓰지 않도록 `prepare_data.py`와 테스트에서 확인했습니다.

### 데이터 분할

```text
Batch 1, 41 cells
  ├─ 5-fold × 10-repeat CV: 모델 안정성 확인
  ├─ charge-policy Group CV: 보지 못한 정책의 민감도 확인
  └─ 전체 41 cells: 최종 모델 재학습

Batch 2, 43 cells
  └─ 최종 모델·하이퍼파라미터 확정 후 평가
     후보의 사후 비교 결과는 최종 선택과 분리
```

최종 모델은 Batch 2 성능을 확인하기 전에 확정했습니다. Batch별 분포와 후보의 사후 성능도 함께 보고하되, 모델·하이퍼파라미터 선택은 Batch 1에서 정한 기준으로 유지했습니다.

Batch 3는 선택 과제입니다. 현재 compact table의 셀 매핑이 저자의 공식 제외 목록과 완전히 일치하지 않아 성능표에서 제외했습니다. 원본 `.mat`에서 추출한 뒤 cell ID를 대조해 평가할 계획입니다.

### 누수 방지

- 결측치 대치와 scaling은 sklearn Pipeline 안에서 각 CV 학습 fold에만 fit합니다.
- `cycle_life`, `batch`, `partition`, `cell_id`는 입력 피처로 사용할 수 없게 차단했습니다.
- 전체 수명곡선으로 계산한 knee는 입력에서 제외했습니다.
- 충전정책 변수는 본 모델 선택과 별도의 민감도 분석으로 나눴습니다.
- Batch 2 결과를 확인한 뒤 후보·하이퍼파라미터를 변경하지 않았습니다.

### 핵심 피처

| 그룹 | 피처 | 의미 |
| --- | --- | --- |
| ΔQ(V) | `log_var_dq`, `log_min_dq`, `dq_mean`, `dq_skew`, `dq_kurt` | 초기 방전곡선 형태의 변화입니다. |
| Capacity | `cap_c2`, `cap_c100`, `cap_diff_100_2` | 초기 용량과 변화량입니다. |
| Degradation trend | `cap_slope_2_100`, `cap_intercept_2_100` | 2–100사이클의 열화 추세입니다. |
| Sensor 후보 | 온도·내부저항 **7개** | 방전 피처에 추가되는 정보를 확인합니다. |
| Policy ablation | `cc1`, `cc2`, `q1_pct`, `avg_charge_time` | 실험 처치 정보의 영향을 확인합니다. |

### 선택 규칙

1. Batch 1 반복 CV MAPE가 최저값에서 **1%p** 이내인 후보를 남깁니다.
2. 남은 후보 중 충전정책 Group CV MAPE가 가장 낮은 모델을 고릅니다.
3. 모델·하이퍼파라미터를 확정하고 Batch 2를 평가합니다.

최종 Elastic Net은 `alpha=0.00464`, `l1_ratio=0.7`입니다. 표본 수 대비 피처가 많고 서로 상관돼 L1·L2 규제를 함께 사용하는 구성을 택했습니다.

```text
연장 셀 병합·데이터 계약 확인
             ↓
cycle 2–100 피처 준비
             ↓
Batch 1 모델·하이퍼파라미터 선택
             ↓
Batch 2 고정 평가
             ↓
성능표·셀별 예측·오류·분포 차이·예측구간 저장
```

구현 위치: [`src/preprocess.py`](src/preprocess.py), [`src/train.py`](src/train.py), [`src/evaluate.py`](src/evaluate.py), [`scripts/run_pipeline.py`](scripts/run_pipeline.py), [`tests/`](tests/)

</details>

<details>
<summary>전체 성능표와 원논문·과제 기준 비교</summary>

### 후보 모델

| 모델 | 피처 | Batch 1 반복 CV MAPE | 정책 Group CV | Batch 2 MAPE | 판단 |
| --- | --- | ---: | ---: | ---: | --- |
| Median baseline | 기준값 | 23.33% | 22.96% | 86.87% | 최소 기준입니다. |
| Variance Ridge | ΔQ 분산 1개 | 9.17% | 8.92% | 32.43% | 논문 단일 피처 기준입니다. |
| Discharge Ridge | 방전 10개 | 8.14% | 9.17% | 24.68% | 규제 선형 비교입니다. |
| **Discharge Elastic Net** | **방전 10개** | **7.49%** | **8.21%** | **26.59%** | **최종 모델입니다.** |
| Random Forest | 방전 10개 | 10.27% | 11.82% | 40.79% | 외부 평가 오차가 컸습니다. |
| Sensor Ridge | 방전+센서 17개 | 11.29% | 12.84% | 22.55% | 사후 비교로 보고했습니다. |
| Policy-aware Elastic Net | 센서+정책 21개 | 8.56% | 10.43% | 25.48% | 최종 선택에서 제외했습니다. |

Sensor Ridge는 Batch 2에서 최종 모델보다 낮은 MAPE를 보였습니다. 이 결과로 모델을 교체하면 Batch 2를 validation set으로 사용하게 되므로, Batch 1에서 선택한 `discharge_elasticnet`을 유지했습니다.

### 최종 모델

| 평가 항목 | 결과 |
| --- | ---: |
| Batch 1 repeated CV MAPE | 7.49% ± 2.19% |
| Batch 1 policy Group CV MAPE | 8.21% |
| **Batch 2 MAPE** | **26.59%** |
| Batch 2 Median APE | 22.95% |
| Batch 2 MAE | 115.78 cycles |
| Batch 2 RMSE | 129.34 cycles |
| Batch 2 R² | -1.90 |
| MAPE bootstrap 95% CI | 21.19–34.02% |

### 원논문·과제 기준과의 GAP

| 비교 기준 | 논문 수치 | 본 프로젝트 | GAP | 비교 조건 |
| --- | ---: | ---: | ---: | --- |
| 과제 지정 headline test error | 9.1% | 26.59% | **+17.49%p** | 과제 요구 기준입니다. |
| Table 1 discharge / primary / all cells | 13.0% | 26.59% | **+13.59%p** | 현재 모델과 가장 가까운 기준입니다. |
| Table 1 full / primary / all cells | 14.1% | 26.59% | +12.49%p | 피처 구성이 다릅니다. |
| Table 1 full / anomaly 제외 | 7.5% | 26.59% | 직접 비교하지 않습니다. | 이상 셀 제외 조건이 다릅니다. |

9.1%, 13.0%, 7.5%는 서로 다른 평가 조건의 수치입니다. 과제에서 요구한 9.1% GAP을 보고하고, Batch 2 전체 셀을 평가한 13.0%도 함께 비교했습니다.

Batch 2 R² -1.90은 해당 평가셋의 평균 수명 예측보다 제곱오차가 컸다는 뜻입니다. 두 Batch의 타깃·입력 분포 차이와 외부 평가의 성능 저하를 함께 확인했지만, 분포 이동 하나의 영향으로 원인을 분리하지는 않았습니다.

근거 파일: [`results/model_selection.csv`](results/model_selection.csv), [`results/model_performance.csv`](results/model_performance.csv), [`results/uncertainty.json`](results/uncertainty.json)

</details>

<details>
<summary>오류·분포 차이·예측구간 분석</summary>

### 오류가 큰 셀

Batch 2의 실제 평균은 **473.2 cycles**, 예측 평균은 **586.2 cycles**였습니다. 평균적으로 약 **113 cycles**를 높게 예측했습니다.

| Cell | 실제 | 예측 | APE |
| --- | ---: | ---: | ---: |
| `b2c1` | 148 | 368 | 148.8% |
| `b2c20` | 502 | 791 | 57.6% |
| `b2c42` | 466 | 665 | 42.8% |

- 수명 500 미만 셀 MAPE는 **28.15%**였습니다.
- 수명 500 이상 셀 MAPE는 **21.41%**였습니다.
- `b2c1`을 제외한 민감도 MAPE는 **23.68%**였습니다.

`b2c1`을 제외하면 점수는 좋아졌지만 기본 성능표에서는 유지했습니다. 원논문 평가에 포함된 셀이며 측정 오류의 근거를 확인하지 못했기 때문입니다.

### 외부 평가와 함께 확인한 조건

1. **타깃 분포:** 중앙값이 Batch 1의 842에서 Batch 2의 481 cycles로 달랐습니다.
2. **입력 피처 분포:** SMD는 `dq_mean` **-1.63**, `cap_diff_100_2` **-1.47**, `cap_c100` **-1.41**이었습니다.
3. **작은 실제값:** MAPE는 수명이 짧은 셀의 오차를 크게 반영합니다.
4. **표본 크기:** 41개 셀에서 피처·하이퍼파라미터를 선택해 추정 변동이 클 수 있습니다.
5. **실험 조건:** 충전정책·수집 시점·Batch가 독립적으로 달라지지 않습니다.

명목 **90%** conformal prediction interval의 Batch 2 실제 coverage는 **83.72%**였습니다. Batch 1 잔차로 만든 구간이 다른 Batch에서 같은 coverage를 유지하는지도 확인해야 합니다.

</details>

<details>
<summary>ESS 적용 가설과 다음 검증</summary>

이 실험은 한 종류의 LFP/graphite 셀과 실험실 급속충전 조건에 한정됩니다. calendar aging, 주변 온도, SOC window와 제조 lot 차이를 포괄하지 않으므로 실제 BESS 교체·보증 결정에 바로 적용할 수는 없습니다.

다음에는 초기 시험으로 추가 검사가 필요한 셀을 좁히는 선별 용도를 검토하려고 합니다. 팩 조립 전 셀 매칭, 장기 열화시험 우선순위, 빠른 열화 가능성이 있는 셀의 점검 순서, 충전정책별 잔차를 이용한 추가 실험 설계가 후보입니다. 현재 모델로 이런 업무 효과를 측정한 결과는 없습니다.

1. 원본 `.mat`에서 Batch 3를 다시 추출하고 공식 cell ID를 대조합니다.
2. 다른 제조 lot와 운용 조건의 외부 데이터를 평가합니다.
3. leave-one-protocol-out CV로 새로운 충전정책의 성능을 확인합니다.
4. Batch별 calibration과 weighted conformal 방법을 비교합니다.
5. 예측구간을 활용하는 의사결정 기준을 설계합니다.

</details>

<details>
<summary>파일 구성과 참고문헌</summary>

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

- Severson, K. A. et al. (2019). Data-driven prediction of battery cycle life before capacity degradation. *Nature Energy*, 4, 383–391. [Paper](https://doi.org/10.1038/s41560-019-0356-8) · [Code](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation)
- Geslin, A. et al. (2023). Selecting the appropriate features in battery lifetime predictions. *Joule*. [Paper](https://doi.org/10.1016/j.joule.2023.07.021) · [Code](https://github.com/geslina/Joule_2023_Perspective)
- Li, Y. et al. (2024). Predicting battery lifetime under varying usage conditions from early aging data. *Cell Reports Physical Science*. [Paper](https://doi.org/10.1016/j.xcrp.2024.101891)
- Attia, P. M. et al. Battery lifetime knee-point analysis and prediction. [Paper](https://www.sciencedirect.com/science/article/pii/S2666546820300069)

</details>
