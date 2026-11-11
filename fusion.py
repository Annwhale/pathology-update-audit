# +--------------------------------------+
# | Pathology update audit               |
# | Zhang et al.                         |
# +--------------------------------------+
# 固定特征顺序、概率损失与源数据校准。
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
EPS = 1e-12

def summary(p):
    order=np.sort(p,axis=1)
    return np.column_stack([p,-(p*np.log(np.clip(p,1e-12,1))).sum(1),1-p.max(1),order[:,-1]-order[:,-2]])

def post_features(sp,tp,quality):
    x=np.column_stack([summary(sp),summary(tp),tp-sp,quality])
    assert x.shape==(len(sp),43) and np.isfinite(x).all()
    return x

def probability_metrics(p,y,metadata):
    p=np.asarray(p,float);y=np.asarray(y,int)
    assert p.shape==y.shape and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    clipped=np.clip(p,EPS,1-EPS)
    bins=np.minimum((p*15).astype(int),14)
    entries=[];ece=0.;mce=0.
    for k in range(15):
        select=bins==k;n=int(select.sum())
        if n:
            conf=float(p[select].mean());freq=float(y[select].mean());gap=abs(conf-freq)
            ece+=n/len(y)*gap;mce=max(mce,gap)
        else:conf=freq=gap=None
        entries.append(dict(metadata,bin=k,bin_left=k/15,bin_right=(k+1)/15,n=n,mean_probability=conf,observed_TUM_fraction=freq,absolute_gap=gap))
    result=dict(metadata,n=len(y),TUM_n=int(y.sum()),TUM_fraction=float(y.mean()),
      Brier=float(np.mean((p-y)**2)),NLL=float(-np.mean(y*np.log(clipped)+(1-y)*np.log1p(-clipped))),
      AUC=float(roc_auc_score(y,p)),ECE_15_equal_width=ece,MCE_15_equal_width=mce,nonempty_bins=int(len(np.unique(bins))))
    return result,entries

def calibration_fit(p,y):
    x=logit(np.clip(np.asarray(p,float),EPS,1-EPS));y=np.asarray(y,float)
    def objective(v):
        z=v[0]*x+v[1];res=expit(z)-y
        return float(np.mean(np.logaddexp(0,z)-y*z)),np.array([np.mean(res*x),np.mean(res)])
    r=minimize(objective,np.array([1.,0.]),jac=True,method='L-BFGS-B',
               options={'maxiter':10000,'ftol':1e-14,'gtol':1e-10})
    if not r.success:raise RuntimeError('Platt optimizer failed: '+str(r.message))
    def tnll(logt):
        z=x/np.exp(logt);return float(np.mean(np.logaddexp(0,z)-y*z))
    t=minimize_scalar(tnll,bounds=(np.log(.05),np.log(20.)),method='bounded',options={'xatol':1e-12})
    if not t.success:raise RuntimeError('Temperature optimizer failed')
    return {'platt':{'slope':float(r.x[0]),'intercept':float(r.x[1]),'fit_NLL':float(r.fun),'iterations':int(r.nit),
        'success':bool(r.success),'message':str(r.message),'gradient_max_abs':float(np.max(np.abs(r.jac)))},
        'temperature':{'T':float(np.exp(t.x)),'fit_NLL':float(t.fun),'function_evaluations':int(t.nfev),
        'success':bool(t.success),'message':str(t.message),'bounds':[.05,20.]},
        'probability_clip':EPS,'calibration_fit_basis':'HMU archived R-only-to-G heldout posterior_TUM; same historical source/model family; no independent calibration validation'}

def applycal(p,cal):
    x=logit(np.clip(p,EPS,1-EPS))
    return {'q_platt':expit(cal['platt']['slope']*x+cal['platt']['intercept']),
            'q_temperature':expit(x/cal['temperature']['T'])}

FEATURE_COLUMNS = {
    'full': np.arange(43),
    'no_quality': np.arange(33),
    'no_difference': np.r_[0:24, 33:43],
    'probabilities': np.r_[0:9, 12:21],
    'summaries': np.arange(24),
}

def fit_head(features, labels, variant='full', seed=42):
    # 输入为源数据交叉拟合特征；缩放器仅在该输入上拟合。
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    columns = FEATURE_COLUMNS[variant]
    model = make_pipeline(StandardScaler(), LogisticRegression(
        C=.1, class_weight=None, max_iter=2500, tol=1e-5, random_state=seed))
    model.fit(np.asarray(features)[:, columns], labels)
    return model, columns

def update_actions(score, student_action, called, weight=2):
    # 未调用图块保留初始动作；调用后只接受正增益更新。
    score = np.asarray(score, float)
    base = np.asarray(student_action, bool)
    old_risk = np.where(base, 1-score, weight*score)
    gain = old_risk - np.minimum(weight*score, 1-score)
    return np.where(np.asarray(called, bool) & (gain > 1e-12),
                    score >= 1/(1+weight), base)

def error_ledger(reference, initial, final):
    y, before, after = [np.asarray(v, bool) for v in (reference, initial, final)]
    return dict(
        FN=int((y & ~after).sum()), FP=int((~y & after).sum()),
        repaired_FN=int((y & ~before & after).sum()),
        new_FN=int((y & before & ~after).sum()),
        repaired_FP=int((~y & before & ~after).sum()),
        new_FP=int((~y & ~before & after).sum()))

def spatial_errors(action, positive_pixels, negative_pixels):
    a = np.asarray(action, bool)
    pos, neg = np.asarray(positive_pixels), np.asarray(negative_pixels)
    fp, fn = int(neg[a].sum()), int(pos[~a].sum())
    total = int((pos+neg).sum())
    if total <= 0:
        raise ValueError('Scored support is empty')
    return dict(FP=fp, FN=fn, pixels=total,
                area_error_pp=100*abs(fp-fn)/total,
                pixel_error_percent=100*(fp+fn)/total,
                grid_floor_percent=100*np.minimum(pos, neg).sum()/total)
