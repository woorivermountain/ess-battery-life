# DAY 2 평가항목 대응표

| 평가항목 | 배점 | 구현 증거 | 자체 점검 |
|---|---:|---|---|
| 전략 기반 Feature·모델 구현 | 20 | `src/config.py`, `src/models.py`, policy-aware ablation | DAY 1의 ΔQ(V)·용량 추세를 코드의 feature contract로 고정 |
| 개발 파이프라인 | 20 | `scripts/run_pipeline.py`, `src/train.py`, 저장 모델·manifest | 한 명령으로 전처리→선택→평가→그림 생성 |
| 데이터 분할 적절성 | 10 | `src/preprocess.py`, `tests/test_data_contract.py` | B1 41개 개발, B2 43개 잠금 평가, ID 중복·연장셀 검증 |
| 핵심 변수 구현 | 10 | `src/features.py`, `results/batch1_correlations.csv` | ΔQ, 용량 slope/intercept, 센서 후보와 shift 진단 |
| 성능 포맷·목표 GAP | 10 | `results/model_performance.csv`, `RESULTS.md` | MAPE/RMSE/MAE/MedAPE/R², 9.1%와 동조건 13.0% GAP 분리 |
| 성능 해석 | 10 | `results/uncertainty.json`, `error_analysis_top10.csv` | bootstrap CI, conformal coverage, 극단 셀 포함 |
| ESS 도메인 해석 | 10 | `README.md` ESS 운영 관점, `MODEL_CARD.md` | screening·보증·유지보수 용도와 금지 용도 구분 |
| 한계·개선 방향 | 10 | `research/literature_review.md`, `RESULTS.md` | protocol confounding, batch shift, 작은 n, Batch3 audit |

## 제출 전 확인

- [x] 공개 GitHub 저장소
- [x] 우강산 / SKALA 3반 / U088 표기
- [x] README에 목적·구조·설정·EDA·모델링·성능·오류·도메인·참고문헌 포함
- [x] Batch 2 Target-Test 성능 및 논문 대비 GAP
- [x] 3개 notebook과 모듈화된 `src/`
- [x] `requirements.txt`, 재현 명령, 모델 artifact
- [x] GitHub Actions에서 5개 테스트 통과
- [x] 테스트셋 사후 선택 방지 규칙 명시
- [x] 논문 headline·Table 1·이상치 제외 수치의 비교 조건 구분
