# Model card — discharge_elasticnet

- Owner: 우강산 (SKALA 3반, U088)
- Intended use: early screening and experiment prioritization for similar LFP/graphite cells
- Not intended for: direct safety control, warranty decisions without human review, other chemistries
- Training data: Batch 1, 41 cells
- Locked evaluation: Batch 2, 43 cells
- Inputs: 10 discharge-derived features from cycles 2–100
- Target: final cycle life
- Test performance: MAPE 26.59%, RMSE 129.34, MAE 115.78 cycles
- Key risk: severe target/covariate shift and under-coverage on Batch 2
- Monitoring: input-range drift, protocol mix, MAPE/MAE by lifetime band, interval coverage
- Retraining trigger: sustained feature SMD > 0.5 or interval coverage below 85%
