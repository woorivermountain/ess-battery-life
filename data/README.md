# Data provenance

`source/severson_features.csv` is the cell-level early-cycle feature table derived from the
MIT–Stanford battery dataset (Severson et al., 2019). It contains 124 cells: Batch 1 = 41,
Batch 2 = 43, Batch 3 = 40. Raw `.mat` files are not redistributed here.

`scripts/prepare_data.py` applies the official continuation merge for `b1c0`–`b1c4` and
writes `processed/severson_features.csv`. These five cells continued cycling in Batch 2;
using their pre-continuation labels would silently corrupt the target.

Batch 3 is retained for descriptive checks only. Its compact extraction needs a raw-file
cell-mapping audit before it may be reported as an external-test result.
