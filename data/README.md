# 데이터 구성과 전처리 기준

## 파일

- `source/severson_features.csv`: MIT–Stanford Battery Dataset에서 추출한 셀 단위 초기 피처
- `processed/severson_features.csv`: 연장 셀의 최종 수명을 반영한 모델 입력 테이블

총 124개 셀로 구성되며 논문의 최종 partition 기준 셀 수는 Batch 1=41, Batch 2=43,
Batch 3=40이다. 용량이 큰 원본 `.mat` 파일은 저장소에 포함하지 않았다.

## 연장 셀 처리

Batch 1에서 시험이 중단된 `b1c0`–`b1c4`는 Batch 2에서 계속 측정됐다. 두 구간을
연결하지 않으면 장수명 셀의 target이 실제보다 짧아지므로 다음 최종 수명을 적용했다.

| Cell | Cycle Life |
|---|---:|
| `b1c0` | 1852 |
| `b1c1` | 2160 |
| `b1c2` | 2237 |
| `b1c3` | 1434 |
| `b1c4` | 1709 |

`python scripts/prepare_data.py`를 실행하면 위 값을 반영한 processed table을 다시 만든다.
동일한 병합 규칙은 `src/preprocess.py`와 `tests/test_data_contract.py`에서도 확인한다.

## Batch 3 사용 제한

현재 compact table의 Batch 3 cell mapping은 저자 공식 제외 목록과 다시 대조할 필요가
있다. 따라서 Batch 3는 기술통계 확인에만 남겨 두고 모델 성능으로 보고하지 않았다.
원본 파일에서 공식 로더와 같은 기준으로 재추출한 뒤 외부 평가에 사용해야 한다.
