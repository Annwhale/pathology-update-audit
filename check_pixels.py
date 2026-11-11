# +--------------------------------------+
# | BSPC - Zhang et al. - v28             |
# | Reproducible analysis source         |
# +--------------------------------------+
# 运行前核对输入路径；关键参数和数据映射以随附配置为准。
"""Repeat independent verification in an isolated temporary copy; preserve all inputs."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys,tempfile
source=Path(sys.argv[1]).resolve()/'reproducibility' if len(sys.argv)>1 else Path(__file__).resolve().parent
with tempfile.TemporaryDirectory(prefix='bspc_v18_replay_') as td:
 work=Path(td)/'replay';shutil.copytree(source,work,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 previous=work/'reports/bspc_v18_discovery_20261022/roi_analysis/independent_replay_356.json'
 if previous.exists():previous.unlink()
 script=work/'analysis/356_bspc_ROI_independent_pixel_replay_20261022.py'
 result=subprocess.run([sys.executable,str(script),'--verify'],cwd=work,text=True,capture_output=True)
 if result.returncode:
  print(result.stdout[-2000:]);print(result.stderr[-4000:],file=sys.stderr);sys.exit(result.returncode)
 report=json.loads(previous.read_text());assert report['mismatches']==0
 print(json.dumps({k:report[k] for k in ['status','mismatches','ROIs','tiles','control_outcome_rows','baseline_rows','reference_tile_rows','dense_pixel_checks']},indent=2))
