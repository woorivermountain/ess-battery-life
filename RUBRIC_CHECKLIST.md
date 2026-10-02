# DAY 2 평가항목 확인표

채점 시 구현 근거를 빠르게 찾을 수 있도록 평가표와 저장소 파일을 연결했다.

## 1. 전략 → 구현 반영 (20점)

| 확인 내용 | 구현 근거 |
|---|---|
| DAY 1의 ΔQ(V) 분석을 실제 피처로 구현 | `src/config.py`의 `DISCHARGE_FEATURES`, `src/features.py` |
| 용량 감소 추세 구현 | `cap_c2`, `cap_c100`, `cap_diff_100_2`, slope, intercept |
| 작은 표본·공선성을 고려한 모델 | Ridge, Elastic Net을 주 후보로 구현 |
| 정책 정보와 건강 신호 분리 | policy-blind 최종 후보와 policy-aware ablation 분리 |
| EDA와 모델 선택 논리 연결 | `README.md` 1장 EDA→구현 표 |

## 2. Pipeline 개발 (40점)

| 확인 내용 | 구현 근거 |
|---|---|
| 실행 가능한 전체 pipeline | `scripts/run_pipeline.py`, `src/train.py` |
| Batch 1 / Batch 2 분리 | `make_course_split()`, `run_manifest.json` |
| 공식 셀 수 검증 | Batch 1=41, Batch 2=43, Batch 3=40 계약 검사 |
| 연장 셀 target 병합 | `b1c0`–`b1c4` 공식 수명 반영 및 테스트 |
| 식별자·target 누수 차단 | `validate_feature_set()` |
| 전처리 누수 방지 | imputer와 scaler를 sklearn Pipeline 안에서 fit |
| 모델 선택 안정성 | 5-fold×10-repeat CV와 charge-policy Group CV |
| 핵심 변수 구현 | ΔQ(V), 용량 추세, 온도·IR 후보, 정책 ablation |
| 재현 가능한 환경 | `requirements.txt`, 고정 seed, 실행 명령 |
| 자동 검증 | GitHub Actions에서 pytest 5개 통과 |

## 3. 성능 리포팅 및 GAP 해석 (20점)

| 확인 내용 | 구현 근거 |
|---|---|
| 기본 성능표 | `results/model_performance.csv` |
| 모델 선택 과정 | `results/model_selection.csv` |
| 셀별 예측·오류 | `results/batch2_predictions.csv`, `error_analysis_top10.csv` |
| 과제 목표 9.1% GAP | +17.49%p 보고 |
| 동조건 논문 성능 비교 | Table 1 discharge/primary/all cells 13.0% 대비 +13.59%p |
| 여러 지표 보고 | MAPE, Median APE, MAE, RMSE, R² |
| 불확실성 보고 | bootstrap 95% CI, conformal interval coverage |
| 이상 셀 민감도 | `b2c1` 포함 기본값과 제외값을 구분 |

## 4. 도메인 해석 및 한계 (20점)

| 확인 내용 | 구현 근거 |
|---|---|
| ESS 활용 범위 | 초기 셀 선별, 시험 우선순위, 점검 순서로 한정 |
| 적용하면 안 되는 범위 | 안전제어·자동 보증판단·타 화학계 직접 적용 금지 |
| Batch shift 정량화 | target 중앙값과 feature SMD 보고 |
| 정책 confounding 해석 | Geslin et al. 근거와 policy-aware ablation |
| 외부평가 오류 해석 | 평균 +113 cycles 과대예측, 단수명 구간 오차 증가 |
| 배포 전 요구사항 | 외부 lot, 온도, SOC, drift, calibration 검증 |
| Batch 3 한계 | cell mapping 재감사 전 성능 미보고 |

## 제출 상태

- [x] 공개 GitHub 저장소
- [x] SKALA 3반 우강산 U088 표기
- [x] README에 목적, EDA, 피처, 모델, 성능, 오류, 도메인 해석 포함
- [x] Batch 2 Target-Test 성능과 논문 대비 GAP 포함
- [x] 3개 notebook과 모듈화된 `src/` 제공
- [x] `requirements.txt`와 재현 명령 제공
- [x] 최종 모델과 결과 CSV 저장
- [x] GitHub Actions 테스트 통과
- [x] 논문 headline, Table 1, 이상 셀 제외 수치의 조건 구분
