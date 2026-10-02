# 문헌 비교와 재현성 감사

## 논문별로 무엇을 가져왔는가

| 연구 | 데이터/평가 | 핵심 기여 | 이 프로젝트에 반영 | 그대로 쓰면 생기는 허점 |
|---|---|---|---|---|
| Severson et al. 2019 | LFP 124 cells, B1/B2/B3 | 초기 ΔQ(V)로 수명 예측 | ΔQ 분산 기준선, 로그 타깃, Elastic Net | 9.1%는 모든 표의 Batch 2 all-cell 수치가 아님; anomaly 제외 수치와 혼용 위험 |
| Geslin et al. 2023 | 같은 공개 데이터 재감사 | 충전 피처가 처치 조건을 인코딩함을 지적 | policy-blind 최종 모델 + policy-aware ablation | 무작위 split이면 같은 프로토콜이 양쪽에 있어 generalization을 과대평가 |
| Li et al. 2024 | NMC 225 cells, 다양한 usage | in/out-of-distribution 성능과 계층 모델 | Batch shift를 별도 문제로 해석 | 더 큰 NMC 결과를 41개 LFP 표본에 직접 이식할 수 없음 |
| Attia et al. 2022 | knee/onset 분석 | knee와 예측 불확실성 | knee를 오류 분석 후보로 두고 예측구간 보고 | knee는 초기 100사이클 안에 없을 수 있어 입력 피처로 직접 쓰기 어려움 |
| Zhang et al. 2025 | 다양한 aging trajectory | 대규모 표현학습 가능성 | 향후 확장안으로만 제시 | n=41에서 deep learning은 성능보다 분산·누수 위험이 큼 |

## 9.1%를 그대로 목표로 두면 생기는 비교 오류

Severson 논문의 Table 1은 모델과 test partition을 나눠 보고한다. Primary test의
all-cell MAPE는 variance 14.7%, discharge 13.0%, full 14.1%다. full model의
7.5%는 rapidly-failing anomaly 한 셀을 제외한 괄호 값이다. 반면 과제의 9.1%는
논문 abstract의 headline test error다. 따라서 다음 셋을 분리해야 한다.

1. 과제 요구 충족용: `26.59 - 9.1 = +17.49%p`
2. 같은 primary/all-cell에 가까운 비교: `26.59 - 13.0 = +13.59%p`
3. anomaly 제외 민감도: 기본 성능표가 아니라 별도 분석

평가가 불리해져도 `b2c1`을 기본표에서 제거하지 않은 이유다.

## 충전 프로토콜은 피처인가, 누수인가

정답은 사용 목적에 따라 다르다. 아직 운용 정책이 정해지지 않은 셀의 **고유 건강도**를
스크리닝한다면 충전 정책은 처치 정보이므로 제외해야 한다. 이미 운용 계획이 정해졌고
그 정책 아래의 수명을 예측한다면 알려진 공변량으로 사용할 수 있다. 본 과제는 전자를
주 사용례로 정하고 후자는 ablation으로만 보고했다.

이번 결과에서도 policy-aware 모델은 내부 CV 8.56%지만 Batch 2 25.48%였다.
정책 변수를 추가했다고 cross-batch 문제가 해결되지 않았다. 반대로 policy-only 모델을
최종 후보로 삼지 않은 것은 프로토콜을 외워 얻은 점수를 셀 상태 추론으로 오해하지 않기 위해서다.

## 추가로 확인한 논리적 비약

- **상관 → 원인**: ΔQ(V)와 수명이 강하게 상관돼도 열화 메커니즘의 인과 증명은 아니다.
- **내부 CV → 현장 성능**: B1 CV 7.49%가 B2 26.59%로 악화됐다. 같은 batch 내 무작위
  CV만으로 production readiness를 주장할 수 없다.
- **복잡도 → 성능**: Random Forest의 train MAPE는 3.51%지만 B2는 40.79%다.
- **MAPE 하나 → 충분한 평가**: 짧은 수명 셀에 큰 가중치가 생긴다. RMSE, MAE,
  Median APE, R²와 불확실성을 함께 봐야 한다.
- **예측구간 → 신뢰 가능**: nominal 90% 구간의 외부 coverage가 83.72%였다.
  shift 아래에서는 calibration도 다시 검증해야 한다.

## 원문과 코드

- [Severson et al., Nature Energy 2019](https://web.mit.edu/braatzgroup/Severson_NatureEnergy_2019.pdf)
- [Severson official repository](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation)
- [Geslin et al., Joule 2023](https://www.cell.com/joule/fulltext/S2542-4351(23)00319-7)
- [Geslin feature-selection code](https://github.com/geslina/Joule_2023_Perspective)
- [Li et al., CRPS 2024](https://www.cell.com/cell-reports-physical-science/fulltext/S2666-3864(24)00127-9)
- [Knee-point analysis](https://www.sciencedirect.com/science/article/pii/S2666546820300069)
- [Uncertainty-aware early prediction](https://pubs.rsc.org/en/content/articlelanding/2023/dd/d2dd00067a)
- [Cross-dataset BatteryLife benchmark](https://arxiv.org/html/2502.18807v5)
- [Cross-diverse aging trajectories, Nature Machine Intelligence 2025](https://www.nature.com/articles/s42256-024-00972-x)
