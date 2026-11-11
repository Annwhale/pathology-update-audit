# +--------------------------------------+
# | BSPC - Zhang et al. - v28             |
# | Reproducible analysis source         |
# +--------------------------------------+
# 运行前核对输入路径；关键参数和数据映射以随附配置为准。
"""Portable score-to-endpoint replay. Requires numpy/pandas; no encoders or image download.
Pass the v22 submission folder. Raw-image validation has separate dense-pixel receipts.
"""
from pathlib import Path
import hashlib,json,sys
import numpy as np,pandas as pd

if len(sys.argv) < 2:
 raise SystemExit('Usage: python replay.py /path/to/research-package')
P=Path(sys.argv[1]).resolve();V=P/'reproducibility/v22';D=P/'evidence/v22';checked={'patch':0,'originalROI':0,'BCSS':0}
def scores(z):
 s,t=z['student_probability'][:,8],z['teacher_probability'][:,8];q=z['q_original'];a={k:z[k]for k in z.files if k.startswith('q_')};a.update(teacher=t,mean=(s+t)/2,bounded=np.clip(q,np.minimum(s,t),np.maximum(s,t)),student_threshold=s);return s,a
def action(score,base,call,w):
 r0=w*score;r1=1-score;old=np.where(base,r1,r0);a=score>=1/(1+w);return np.where(call&(old-np.minimum(r0,r1)>1e-12),a,base)
patch=pd.read_csv(D/'ablation_endpoints/patch_all_matched_endpoint_results.csv').set_index(['source','teacher','seed','setting','mode','weight','control'])
for source in['yunnan','tcga','guangdong']:
 for path in sorted((V/'scores/patch'/source).glob('*.npz')):
  z=np.load(path);teacher=str(z['teacher_name']);seed=int(z['seed']);s,q=scores(z);base=s>=.5;y=z['reference_TUM'].astype(bool)
  for setting,call in enumerate(z['call_masks']):
   for w in[1,2,5,10]:
    for mode,decisionw in[('fixed_2to1_actions_reweighted',2),('matched_call_redecision_at1/(1+w)',w)]:
     for control,score in q.items():
      a=action(score,base,call.astype(bool),decisionw);fn=int((y&~a).sum());fp=int((~y&a).sum());r=patch.loc[(source,teacher,seed,setting,mode,w,control)];assert fn==r.FN and fp==r.FP and abs(w*fn+fp-r.weighted_loss)<1e-8;checked['patch']+=1
oldcounts=pd.read_csv(P/'reproducibility/reports/bspc_v19_expanded_20261022/geometric_diagnostic/reference_tile_geometry_all4860.csv').sort_values('tile_id');bcsscounts=pd.read_csv(D/'bcss_spatial/native_reference_tile_counts.csv').sort_values('row_position')
for kind,paths,counts,table,idname in[
 ('originalROI',V/'scores/roi',oldcounts,D/'ablation_endpoints/ROI_all_endpoint_results.csv','roi'),
 ('BCSS',V/'bcss_spatial/predictions',bcsscounts,D/'bcss_spatial/all_ROI_endpoint_results.csv','region_id')]:
 result=pd.read_csv(table).set_index([idname,'teacher','seed','budget','weight','control']);pos=counts.positive_pixels.to_numpy(np.int64);neg=counts.negative_pixels.to_numpy(np.int64)
 if kind=='BCSS':geometry=pd.read_csv(V/'bcss_spatial/prepared/image_only_tile_manifest.csv');ids=geometry.region_id.to_numpy();tids=geometry.tile_id.to_numpy()
 for path in sorted(paths.glob('*.npz')):
  z=np.load(path);teacher,seed=path.stem.rsplit('_seed',1);seed=int(seed);s,q=scores(z);base=s>=.5;v=np.clip(s,1e-12,1-1e-12);entropy=-(v*np.log(v)+(1-v)*np.log(1-v))
  if kind=='originalROI':ids=z['roi'];tids=z['tile_id'];assert np.array_equal(oldcounts.tile_id,tids)and np.array_equal(oldcounts.roi,ids)
  assert len(ids)==len(s)==len(pos)
  for roi in np.unique(ids):
   idx=np.flatnonzero(ids==roi);pp=pos[idx];nn=neg[idx];total=int((pp+nn).sum())
   if not total:continue
   order=np.lexsort((tids[idx],-entropy[idx]))
   for budget in[.1,.3,.5,1.]:
    call=np.zeros(len(idx),bool);call[order[:int(np.floor(len(idx)*budget))]]=True
    for w in[1,2,5,10]:
     for control,score in q.items():
      a=action(score[idx],base[idx],call,w);tp=int(pp[a].sum());fp=int(nn[a].sum());fn=int(pp[~a].sum());tn=int(nn[~a].sum());r=result.loc[(roi,teacher,seed,budget,w,control)];assert[tp,fp,fn,tn]==[r.TP,r.FP,r.FN,r.TN];assert abs(100*abs(fp-fn)/total-r.area_error_pp)<1e-9 and abs(100*(fp+fn)/total-r.pixel_error_percent)<1e-9;checked[kind]+=1
receipt={'status':'PASS_PORTABLE_SAVED_SCORE_TO_ENDPOINT_REPLAY','rows':checked,'scope':'Reconstruct actions and integer outcomes from saved probabilities/call masks and source-native reference counts. This is not independent encoder retraining or new clinical validation.','package':str(P)}
print(json.dumps(receipt,indent=2))
if len(sys.argv)>2:Path(sys.argv[2]).write_text(json.dumps(receipt,indent=2)+'\n')
