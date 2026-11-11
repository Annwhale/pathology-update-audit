# +--------------------------------------+
# | BSPC - Zhang et al. - v28             |
# | Reproducible analysis source         |
# +--------------------------------------+
# 运行前核对输入路径；关键参数和数据映射以随附配置为准。
"""Recompute fixed-grid diagnostic in an isolated copy and compare exact count rows."""
from pathlib import Path
import json,shutil,subprocess,sys,tempfile
import pandas as pd
source=Path(sys.argv[1]).resolve()/'reproducibility' if len(sys.argv)>1 else Path(__file__).resolve().parent
rel=Path('reports/bspc_v19_expanded_20261022/geometric_diagnostic')
expected=pd.read_csv(source/rel/'geometric_decomposition_all7200.csv')
with tempfile.TemporaryDirectory(prefix='bspc_v19_geometry_') as td:
 work=Path(td)/'replay';shutil.copytree(source,work,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 p=subprocess.run([sys.executable,str(work/'analysis/375_bspc_labelaware_grid_diagnostic_20261022.py')],cwd=work,text=True,capture_output=True)
 if p.returncode:
  print(p.stderr[-4000:],file=sys.stderr);sys.exit(p.returncode)
 found=pd.read_csv(work/rel/'geometric_decomposition_all7200.csv');pd.testing.assert_frame_equal(expected,found,check_exact=True)
 receipt=json.loads((work/rel/'geometric_diagnostic_complete.json').read_text());assert receipt['identity_failures']==0
 print(json.dumps({k:receipt[k]for k in ['status','ROIs','tiles','integer_identity_rows','identity_failures','dense_pixel_checks','floor_ROI_mean']},indent=2))
