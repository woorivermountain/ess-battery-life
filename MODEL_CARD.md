# 모델 카드: Discharge Elastic Net

## 기본 정보

| 항목 | 내용 |
|---|---|
| 개발자 | 우강산, SKALA 3반 U088 |
| 목적 | 초기 100사이클 신호를 이용한 Cycle Life 회귀 예측 |
| 학습 데이터 | MIT–Stanford Batch 1, 41 cells |
| 평가 데이터 | Batch 2, 43 cells |
| 입력 | ΔQ(V) 통계량과 방전용량 추세 10개 |
| 출력 | 예상 Cycle Life |
| 알고리즘 | 로그 타깃 Elastic Net |
| 하이퍼파라미터 | `alpha=0.00464`, `l1_ratio=0.7` |

## 성능

- Batch 1 repeated CV MAPE: 7.49% ± 2.19%
- Batch 2 MAPE: 26.59%
- Batch 2 MAE: 115.78 cycles
- Batch 2 RMSE: 129.34 cycles
- MAPE bootstrap 95% CI: 21.19–34.02%
- 90% prediction interval coverage: 83.72%

## 사용할 수 있는 범위

동일하거나 유사한 LFP/graphite 셀의 초기 스크리닝과 장기 시험 우선순위를 정할 때
참고값으로 사용할 수 있다. 예측값 하나로 결론을 내리기보다 예측구간과 원시 진단 신호를
함께 확인해야 한다.

## 사용하면 안 되는 범위

- BMS의 실시간 안전제어
- 사람의 검토가 없는 자동 보증 또는 폐기 결정
- 별도 검증 없이 다른 화학계나 제조사의 셀에 적용
- 현재 결과를 셀 열화 메커니즘의 인과 증거로 해석

## 확인된 위험

- Batch 1과 Batch 2의 target과 입력 피처 분포가 모두 다르다.
- 실제 수명이 짧은 셀에서 오차가 커진다.
- Batch 2에서 평균 112.98 cycles를 높게 예측했다.
- Batch 1에서 만든 90% 예측구간의 Batch 2 coverage는 83.72%였다.
- 학습 셀이 41개라 하이퍼파라미터와 계수의 변동 가능성이 크다.

## 운영 시 감시할 항목

- 입력 피처의 Batch 1 범위 이탈률과 standardized mean difference
- 수명 구간별 MAPE와 MAE
- 제조 lot와 충전정책별 잔차
- 예측구간 coverage
- 과대예측과 과소예측 비율

재보정과 재학습의 수치 기준은 별도의 운영 데이터로 정해야 한다. 현재 단계에서는 학습
범위를 벗어난 입력이 늘거나 예측구간 coverage가 계속 낮아지는지를 우선 경보 조건으로 둔다.
