# ESS 배터리 수명 예측

> SKALA 3반 · 우강산 · U088
> MIT–Stanford 배터리의 100사이클 이내 신호로 최종 Cycle Life를 예측한다.

## 결론부터

Batch 1만 사용해 모델과 하이퍼파라미터를 확정한 뒤 Batch 2를 한 번 평가했다.
최종 모델은 **방전 진단 피처 10개 + Elastic Net**이다.

| 검증 구간 | MAPE | RMSE | MAE | 해석 |
|---|---:|---:|---:|---|
| Batch 1 반복 CV | 7.49% ± 2.19% | - | - | 내부 재현성 |
| 충전 프로토콜 그룹 CV | 8.21% | - | - | 보지 못한 프로토콜 민감도 |
| **Batch 2 잠금 평가** | **26.59%** | **129.34 cycles** | **115.78 cycles** | 실제 제출 성능 |

Batch 2 MAPE의 95% bootstrap CI는 **21.19–34.02%**다. 논문 헤드라인
9.1%와의 차이는 +17.49%p지만, 논문 Table 1의 동조건에 가까운 `discharge model /
primary test / all cells` 13.0%와 비교하면 +13.59%p다. 두 기준을 섞지 않았다.

## 프로젝트 개요

- 데이터셋: MIT–Stanford Battery Dataset, LFP/graphite 124 cells
- 개발 데이터: Batch 1, 41 cells
- Target-Test: Batch 2, 43 cells
- 태스크: Regression — `cycle_life` 예측
- 입력 범위: cycle 2–100에서 계산한 방전 용량, ΔQ(V), 온도·내부저항 후보
- 원칙: Batch 2는 모델 선택에 사용하지 않음

## 재현

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

동일 seed와 고정 split을 사용한다. `results/run_manifest.json`은 Batch 2를 선택 후에만
열었다는 실험 계약과 선택 모델을 기록한다.

## EDA에서 모델로 이어진 판단

### 1. Cycle Life

Batch 1 중앙값은 842 cycles, Batch 2는 481 cycles다. Batch 2가 훨씬 짧고 좁아
동일 분포 가정이 깨진다. 무작위 hold-out 대신 **Batch 1 → Batch 2 외부 평가**를 유지했다.

### 2. 열화와 ΔQ(V)

Batch 1에서 `log_var_dq`와 수명의 Spearman 상관은 -0.882다. `log_min_dq`는
-0.854, `dq_mean`은 0.836이다. 따라서 논문의 단일 분산 피처를 기준선으로 두고,
ΔQ(V) 형태와 2–100사이클 용량 추세를 합친 10개 방전 피처를 주 모델 입력으로 삼았다.

### 3. 충전 조건

충전 C-rate는 실험에서 부여한 처치다. 수명과 관련은 있지만, 셀의 초기 건강 신호와
같은 종류의 정보가 아니다. `cc1`, `cc2`, `q1_pct`, 평균 충전시간은 **정책 인코딩
민감도 분석**에만 넣고 최종 후보에서 제외했다. 이 구분이 없으면 높은 점수가 새로운
셀의 건강을 읽은 결과인지, 실험 프로토콜을 외운 결과인지 알 수 없다.

### 4. 공선성과 작은 표본

ΔQ(V) 통계량끼리 강한 상관이 있고 학습 셀이 41개뿐이다. 비선형 대형 모델보다
로그 타깃 + RobustScaler + Elastic Net을 우선했다. Random Forest는 Batch 1 적합
MAPE 3.51%였지만 Batch 2에서 40.79%로 악화되어 과적합 신호가 분명했다.

## 모델 선택

모든 전처리는 CV fold 안에서 fit한다. 5-fold × 10-repeat CV로 후보를 평가한 후,
최저 평균 MAPE에서 1%p 이내인 모델 중 충전 프로토콜 그룹 CV가 가장 낮은 모델을
선택했다. 이 규칙은 Batch 2 결과를 보기 전에 적용했다.

| 후보 | 피처 | 반복 CV MAPE | 프로토콜 그룹 CV | Batch 2 MAPE | 역할 |
|---|---|---:|---:|---:|---|
| Median | 기준선 | 23.33% | 22.96% | 86.87% | 최소 기준 |
| Variance Ridge | ΔQ 분산 1개 | 9.17% | 8.92% | 32.43% | 논문 기준선 |
| Discharge Ridge | 방전 10개 | 8.14% | 9.17% | 24.68% | 선형 후보 |
| **Discharge Elastic Net** | 방전 10개 | **7.49%** | **8.21%** | **26.59%** | 최종 |
| Random Forest | 방전 10개 | 10.27% | 11.82% | 40.79% | 비선형 대조 |
| Sensor Ridge | 방전+센서 17개 | 11.29% | 12.84% | 22.55% | 사후 challenger |
| Policy-aware Elastic Net | 센서+정책 21개 | 8.56% | 10.43% | 25.48% | 선택 제외 ablation |

Sensor Ridge의 Batch 2 점수가 더 좋지만 최종 모델로 바꾸지 않았다. 그렇게 하면
Target-Test를 validation set처럼 쓰게 된다.

## 오류 분석

- `b2c1`은 실제 148 cycles인데 368 cycles로 예측해 APE 148.8%였다. 짧은 수명
  하나가 MAPE를 크게 끌어올리므로 MAE·RMSE·Median APE를 함께 보고했다.
- 선택 피처 중 `dq_mean`의 Batch 간 표준화 평균차는 -1.63, `cap_diff_100_2`는
  -1.47이다. 여러 Batch 2 값이 Batch 1의 1–99% 범위를 벗어난다.
- Batch 1 중앙값 842와 Batch 2 중앙값 481의 차이 때문에 예측이 체계적으로 높다.
- 90% conformal interval의 Batch 2 실제 coverage는 83.72%였다. 구간 폭만 늘리는
  것보다 Batch 이동을 반영한 calibration이 필요하다.

상세 표는 [RESULTS.md](RESULTS.md), 셀별 결과는
[`results/batch2_predictions.csv`](results/batch2_predictions.csv)에 있다.

## ESS 운영 관점

현재 모델은 100사이클 이내의 진단시험으로 셀/모듈의 기대 수명을 빠르게 선별하는
초기 스크리닝 도구다. 보증기간 결정, 팩 조립 전 셀 매칭, 유지보수 우선순위에 활용할
수 있다. 다만 한 종류의 LFP/graphite 셀과 실험실 고속충전 조건에서 얻은 결과이므로
실제 BESS에 바로 배포하면 안 된다. 온도·SOC window·calendar aging·제조 lot가 다른
현장 데이터로 외부 검증하고, 운용 정책별 재보정과 drift monitor를 붙여야 한다.

## 파일 구조

```text
├── data/                       # 출처·가공 데이터와 계약
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
├── src/                        # 전처리·피처·학습·평가·시각화
├── results/                    # 성능표, 예측, 오류·shift·불확실성
├── research/literature_review.md
├── tests/                      # split, leakage, metric, model smoke test
├── requirements.txt
└── README.md
```

## 참고문헌

- Severson et al. (2019), *Nature Energy*. [Paper](https://doi.org/10.1038/s41560-019-0356-8) · [official code](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation)
- Geslin et al. (2023), *Joule*. [Paper](https://doi.org/10.1016/j.joule.2023.07.021) · [code](https://github.com/geslina/Joule_2023_Perspective)
- Li et al. (2024), *Cell Reports Physical Science*. [Paper](https://doi.org/10.1016/j.xcrp.2024.101891)
- Attia et al. (2022), knee-point review and prediction. [Paper](https://www.sciencedirect.com/science/article/pii/S2666546820300069)

논문별 재현 조건과 논리적 주의점은 [문헌 검토](research/literature_review.md)에 정리했다.

평가항목별 구현 위치는 [DAY 2 평가항목 대응표](RUBRIC_CHECKLIST.md)에서 바로 확인할 수 있다.
