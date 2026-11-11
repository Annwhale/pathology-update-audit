# +--------------------------------------+
# | BSPC - Zhang et al. - v28             |
# | Reproducible analysis source         |
# +--------------------------------------+
# 运行前核对输入路径；关键参数和数据映射以随附配置为准。
"""Independent fixed-score replay. No refitting, GPU, image annotation or mutable outputs."""
from pathlib import Path
import hashlib,json,sys
import joblib,numpy as np,pandas as pd
ROOT=Path(sys.argv[1]).resolve()/'reproducibility' if len(sys.argv)>1 else Path(__file__).resolve().parent

def ent(x):
 return -(x*np.log(np.clip(x,1e-12,1))+(1-x)*np.log(np.clip(1-x,1e-12,1)))
def summary(p):
 s=np.sort(p,axis=1)
 return np.column_stack([p,-(p*np.log(np.clip(p,1e-12,1))).sum(1),1-p.max(1),s[:,-1]-s[:,-2]])
def q_from_bundle(b,sp,tp,quality):
 x=np.column_stack([summary(sp),summary(tp),tp-sp,quality]);assert x.shape[1]==43
 return b['posterior'].predict_proba(x)[:,1]
def update(r,s):
 proposed=r>=1/3;risk=np.where(s,1-r,2*r);gain=risk-np.minimum(2*r,1-r)
 return np.where((proposed!=s)&(gain>1e-12),proposed,s)

def main():
 inv=json.loads((ROOT/'patch_replay_inputs/manifest.json').read_text());calls=actions=posteriors=0;maxdiff=0.;pcount=0
 modelbase=ROOT/'experiments/bspc_v10_calibrated_referral_20261014/final_models'
 for row in inv['records']:
  source,name=row['source'],row['name'];path=ROOT/'patch_replay_inputs'/source/f'{name}.npz';assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
  teacher,seed=name.rsplit('_seed',1);b=joblib.load(modelbase/teacher/f'seed{seed}/models.joblib')
  z=np.load(path,allow_pickle=False);sp,tp,q=z['student_probability'],z['teacher_probability'],z['common_posterior_TUM'];s=sp[:,8]>=.5
  qual=np.load(path.parent/'quality.npy');found=q_from_bundle(b,sp,tp,qual);d=float(np.max(abs(q-found)));maxdiff=max(maxdiff,d);assert d<=1e-12;posteriors+=1
  assert np.array_equal(s,z['student']);n=len(s);manifest=pd.read_csv(path.parent/'image_manifest.csv');names=manifest['name'].astype(str).to_numpy();hashed=np.array([hashlib.sha256(('v16-tsr-hardtie|'+v).encode()).hexdigest()for v in names])
  for i,(prior,like,tie,budget)in enumerate(z['settings']):
   if prior=='native':r=sp[:,8]
   else:
    table=b['bayes_tables']['student']['full'if like=='soft'else'diagonal'];u=np.column_stack([1-sp[:,8],sp[:,8]])
    likelihood=u@table if like=='soft'else table[(sp[:,8]>=.5).astype(int)]
    priorvec=np.asarray([.875,.125])if prior=='empirical'else np.asarray([.5,.5]);rp=likelihood*priorvec;r=rp[:,1]/rp.sum(1)
   order=np.lexsort((hashed if tie=='sha256'else np.arange(n),-ent(r)));mask=np.zeros(n,bool);mask[order[:int(n*float(budget))]]=True
   assert np.array_equal(mask,z['call_masks'][i]),(source,name,prior,like,tie,budget,'calls')
   final=np.where(mask,update(q,s),s);assert np.array_equal(final,z['finals'][i]);calls+=1;actions+=1;pcount+=n
 # Confirm ROI fusion from the same43-feature definition as patch predictions.
 e=ROOT/'experiments/bspc_v18_discovery_20261022';qual=np.load(e/'prepared/quality.npy');roi=0
 for path in(e/'predictions').glob('*.npz'):
  teacher,seed=path.stem.rsplit('_seed',1);b=joblib.load(modelbase/teacher/f'seed{seed}/models.joblib');z=np.load(path);q=q_from_bundle(b,z['student_probability'],z['teacher_probability'],qual);d=float(np.max(abs(q-z['fused_probability_TUM'])));assert d<=1e-12;maxdiff=max(maxdiff,d);roi+=1
 print(json.dumps({'status':'PASS_INDEPENDENT_FIXED_SYSTEM_SCORE_CALL_ACTION_REPLAY','patch_probability_vectors_reconstructed':posteriors,'ROI_probability_vectors_reconstructed':roi,'patch_call_masks_reconstructed':calls,'patch_final_action_vectors_reconstructed':actions,'patch_action_elements':pcount,'maximum_fusion_probability_difference':maxdiff,'mismatches':0,'new_fits':0,'new_encoder_inference':0},indent=2))
if __name__=='__main__':main()
