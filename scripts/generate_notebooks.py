"""Write lightweight notebooks that call the tested source modules."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def md(text): return {"cell_type":"markdown","metadata":{},"source":[text]}
def code(text): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[text]}
def write(name, cells):
    payload={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.12"}},"nbformat":4,"nbformat_minor":5}
    (ROOT/"notebooks"/name).write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")

write("01_EDA.ipynb",[
    md("# 01 EDA\nBatch 1 개발 데이터와 잠금 Batch 2의 target/covariate shift를 확인한다."),
    code("from src.preprocess import load_feature_table, make_course_split\nfrom src.features import target_summary, correlation_table\nfrom src.config import DISCHARGE_FEATURES\ndf=load_feature_table(); split=make_course_split(df)\ndisplay(target_summary(df))\ndisplay(correlation_table(split.train, DISCHARGE_FEATURES))"),
    code("import seaborn as sns; import matplotlib.pyplot as plt\nsns.histplot(data=df.query(\"batch in ['b1','b2']\"),x='cycle_life',hue='batch',bins=16,element='step')\nplt.show()")
])
write("02_feature_engineering.ipynb",[
    md("# 02 Feature engineering\nΔQ(V), 용량 추세, 센서, 정책 피처를 구분한다. 정책 피처는 처치 인코딩 ablation에만 사용한다."),
    code("from src.config import FEATURE_SETS\nfrom src.preprocess import load_feature_table, make_course_split\nfrom src.features import feature_shift_table\ndf=load_feature_table(); split=make_course_split(df)\ndisplay(feature_shift_table(split.train, split.target_test, FEATURE_SETS['discharge']))"),
    code("from src.preprocess import validate_feature_set\nvalidate_feature_set(FEATURE_SETS['discharge'])\n# 아래는 명시적으로 허용한 민감도 분석\nvalidate_feature_set(FEATURE_SETS['policy_aware'], allow_policy=True)")
])
write("03_modeling.ipynb",[
    md("# 03 Modeling\nBatch 1에서 모델을 확정한 뒤 저장된 Batch 2 결과를 검토한다."),
    code("import pandas as pd\nselection=pd.read_csv('../results/model_selection.csv')\nperformance=pd.read_csv('../results/model_performance.csv')\ndisplay(selection)\ndisplay(performance.query(\"split == 'target_test_b2'\"))"),
    code("pred=pd.read_csv('../results/batch2_predictions.csv')\ndisplay(pred.head(10))\nprint('90% interval coverage:', pred.covered_90.mean())"),
    md("최종 모델 변경에는 Batch 2 점수를 사용하지 않는다. Sensor Ridge는 더 나은 Batch 2 점수를 보였지만 사후 challenger로만 남긴다.")
])
print("generated 3 notebooks")
