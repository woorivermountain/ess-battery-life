# 모델 성능 및 오류 분석

## 1. 평가 조건

최종 성능을 계산하기 전에 다음 조건을 고정했다.

- Batch 1의 41개 셀만 모델 개발에 사용
- cycle 2–100에서 계산한 피처만 사용
- 5-fold × 10-repeat CV와 충전정책 Group CV로 모델 선택
- Batch 2의 43개 셀은 모델 확정 후 한 번만 평가
- 이상 셀을 임의로 삭제하지 않음
- MAPE 외에 MAE, RMSE, Median APE, R², 신뢰구간을 함께 보고

## 2. 모델 선택 결과

Batch 1 반복 CV에서 `discharge_elasticnet`의 평균 MAPE가 7.49%로 가장 낮았다.
충전정책 Group CV도 8.21%로 후보 중 가장 안정적이었다. 최종 하이퍼파라미터는
`alpha=0.00464`, `l1_ratio=0.7`이다.

Random Forest는 Batch 1 전체를 다시 적합했을 때 MAPE가 3.51%까지 낮아졌지만,
Batch 2에서는 40.79%로 악화됐다. 작은 학습 데이터에서 비선형 모델의 낮은 학습오차를
일반화 성능으로 해석하기 어렵다는 결과다.

Sensor Ridge는 Batch 2에서 22.55%로 가장 낮은 MAPE를 보였다. 다만 이 결과는 최종
평가 이후에 알게 된 값이므로 모델 선택에 사용하지 않았다. Batch 2를 보고 모델을 바꾸면
평가 데이터가 validation 역할을 하게 된다.

## 3. 최종 성능

| 지표 | Batch 2 결과 | 함께 보는 이유 |
|---|---:|---|
| MAPE | **26.59%** | 과제와 논문의 주 평가지표 |
| Median APE | 22.95% | 극단 셀의 영향이 적은 중앙 오차 |
| MAE | 115.78 cycles | 평균적으로 몇 cycle 차이 나는지 표시 |
| RMSE | 129.34 cycles | 큰 오차에 더 큰 가중치 부여 |
| R² | -1.90 | Batch 2 평균 예측 대비 설명력 확인 |

셀 단위 bootstrap 5,000회를 적용한 95% 구간은 다음과 같다.

| 지표 | 95% bootstrap interval |
|---|---:|
| MAPE | 21.19–34.02% |
| MAE | 98.93–133.16 cycles |
| RMSE | 111.33–147.26 cycles |

R²가 음수이므로 현재 모델이 Batch 2에서 충분히 일반화됐다고 볼 수 없다. Batch 1에서
찾은 관계가 존재하더라도 Batch 2의 수명 수준과 피처 범위가 함께 이동하면 예측값의
기준점이 맞지 않을 수 있다.

## 4. 원논문 대비 GAP

| 비교 기준 | 논문 | 현재 결과 | GAP |
|---|---:|---:|---:|
| 과제 지정 headline test error | 9.1% | 26.59% | **+17.49%p** |
| Table 1 discharge / primary / all cells | 13.0% | 26.59% | **+13.59%p** |
| Table 1 full / primary / all cells | 14.1% | 26.59% | +12.49%p |
| Table 1 full / primary / anomaly 제외 | 7.5% | 26.59% | 직접 비교하지 않음 |

과제의 목표값인 9.1%는 논문의 headline 수치다. Table 1의 `discharge model`을 보면
Batch 2에 해당하는 primary test 전체 셀 MAPE는 13.0%다. 7.5%는 full feature model에서
빠르게 실패한 셀 한 개를 제외한 값이다. 따라서 9.1%와의 GAP을 제출하되, 동일 조건에
가까운 비교값도 함께 적었다.

## 5. 오차가 큰 셀

| Cell | 실제 수명 | 예측 수명 | 잔차(실제-예측) | APE |
|---|---:|---:|---:|---:|
| `b2c1` | 148 | 368 | -220 | 148.8% |
| `b2c20` | 502 | 791 | -289 | 57.6% |
| `b2c42` | 466 | 665 | -199 | 42.8% |
| `b2c45` | 487 | 671 | -184 | 37.9% |
| `b2c29` | 498 | 686 | -188 | 37.8% |

Batch 2의 실제 평균은 473.2 cycles, 예측 평균은 586.2 cycles다. 평균적으로
112.98 cycles를 높게 예측했다. 특히 짧은 수명 셀에서 오차가 더 컸다.

| 수명 구간 | MAPE |
|---|---:|
| 실제 수명 `< 500` | 28.15% |
| 실제 수명 `≥ 500` | 21.41% |

`b2c1`을 제외하면 MAPE는 23.68%다. 이 값은 민감도 분석으로만 제시하며 기본 성능은
26.59%를 유지했다. IQR 이상치라는 이유만으로 유효한 평가 셀을 제거하면 모델의 약점을
감출 수 있기 때문이다.

## 6. Batch 이동 점검

| 피처 | Standardized mean difference | Batch 2가 Batch 1의 1–99% 범위를 벗어난 비율 |
|---|---:|---:|
| `dq_mean` | -1.63 | 16.28% |
| `cap_diff_100_2` | -1.47 | 20.93% |
| `cap_c100` | -1.41 | 18.60% |
| `log_min_dq` | 1.12 | 16.28% |
| `dq_skew` | 1.10 | 13.95% |
| `log_var_dq` | 1.09 | 18.60% |

|SMD|가 1을 넘는 피처가 여러 개라는 것은 Batch 2가 단순히 수명만 짧은 것이 아니라
입력 신호의 분포도 다르다는 뜻이다. 최종 모델이 Batch 2를 전반적으로 높게 예측한 현상과
방향이 일치한다.

## 7. 예측 불확실성

Batch 1의 out-of-fold residual로 만든 90% conformal interval 반경은 ±178.07 cycles다.
Batch 2의 실제 coverage는 83.72%로 목표 90%에 못 미쳤다. Batch 1에서 맞춘 예측구간도
분포가 이동한 Batch 2에서는 과신될 수 있음을 확인했다.

다음 단계에서는 Batch별 calibration, weighted conformal prediction, 정책 그룹별
coverage를 비교할 필요가 있다.

## 8. 개선 순서

1. 공식 cell mapping으로 Batch 3를 다시 추출해 완전한 외부 평가를 추가한다.
2. leave-one-protocol-out 검증으로 새로운 충전정책에 대한 이동을 측정한다.
3. `b2c1`의 원시 Q(V), 온도, 내부저항, knee 위치를 확인해 데이터 오류와 실제 조기열화를 구분한다.
4. Batch별 intercept 보정과 계층모델을 비교하되 Batch 2를 다시 선택 데이터로 사용하지 않는다.
5. 다른 제조 lot·온도·SOC 범위의 데이터에서 성능과 interval coverage를 재검증한다.

![내부 검증과 Batch 2 성능](results/figures/02_cv_vs_test.svg)

![Batch 2 실제값과 예측값](results/figures/03_prediction_parity.svg)
