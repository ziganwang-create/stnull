import os
os.environ["OMP_NUM_THREADS"]="3"; os.environ["MKL_NUM_THREADS"]="3"; os.environ["OPENBLAS_NUM_THREADS"]="3"
import json, numpy as np, pandas as pd, scipy.sparse as sp
ROOT=r"D:\project\STImage\prototype"
DATA=os.path.join(ROOT,"data"); RUNS=os.path.join(ROOT,"runs","resnet50")
meta=pd.read_parquet(os.path.join(DATA,"meta.parquet"))
sections=sorted(meta.section.unique())
rng=np.random.default_rng(20260821)

def panel(p):
    return json.load(open(os.path.join(RUNS,"fold_%s"%p,"genes.json"),encoding="utf-8"))
def tf(c):
    rs=c.sum(axis=1,keepdims=True); return np.log1p(1e4*c/np.maximum(rs,1.0))
def pear(A,B):
    A=A-A.mean(0,keepdims=True); B=B-B.mean(0,keepdims=True)
    sa=np.sqrt((A*A).sum(0)); sb=np.sqrt((B*B).sum(0))
    with np.errstate(invalid="ignore",divide="ignore"):
        r=(A*B).sum(0)/(sa*sb)
    r[(sa==0)|(sb==0)]=np.nan; return r

rows=[]
frac_le1=[]; panel_frac=[]
for sec in sections:
    p=sec[0]; pan=panel(p); G=len(pan)
    X=sp.load_npz(os.path.join(DATA,"expr_%s.npz"%sec)).tocsr()
    sg=json.load(open(os.path.join(DATA,"genes_%s.json"%sec),encoding="utf-8"))
    gi={g:i for i,g in enumerate(sg)}
    cols=np.array([gi.get(g,-1) for g in pan]); pres=cols>=0
    n=X.shape[0]; pc=np.zeros((n,G))
    if pres.any(): pc[:,pres]=np.asarray(X[:,cols[pres]].todense())
    pc=np.round(pc)
    lib=np.round(np.asarray(X.sum(1)).ravel())
    ps=pc.sum(1); rest=np.maximum(lib-ps,0.0)
    frac_le1.append(float((pc<=1).mean())); panel_frac.append(float(np.median(ps/np.maximum(lib,1))))
    # --- proper split-half: binomial(count, 0.5) on all 738 categories
    full=np.concatenate([pc,rest[:,None]],axis=1).astype(np.int64)
    h1=rng.binomial(full,0.5); h2=full-h1
    r=pear(tf(h1[:,:G].astype(float)), tf(h2[:,:G].astype(float)))
    rh=float(np.nanmedian(r))
    rows.append((sec,rh, 2*rh/(1+rh) if rh>-1 else np.nan))
df=pd.DataFrame(rows,columns=["section","r_splithalf_med","r_sb_full"])
print(df.to_string(index=False))
print("MEDIAN split-half %.4f  Spearman-Brown-to-full %.4f  range %.4f-%.4f"%(
    df.r_splithalf_med.median(), df.r_sb_full.median(), df.r_splithalf_med.min(), df.r_splithalf_med.max()))
print("frac panel entries count<=1: med %.3f  range %.3f-%.3f"%(np.median(frac_le1),min(frac_le1),max(frac_le1)))
print("panel row-sum / whole-lib: med %.3f range %.3f-%.3f"%(np.median(panel_frac),min(panel_frac),max(panel_frac)))
# de-bias the published plug-in estimator
s=pd.read_csv(os.path.join(ROOT,"analysis_out","ceiling_rtech_rnbr_summary.csv"))
print(s.columns.tolist())
c=[x for x in s.columns if "tech" in x.lower()]
print("plugin r_tech col(s)",c)
v=s[c[0]].to_numpy(float)
rho=2-1.0/np.square(v)
print("plugin r_tech med %.4f range %.4f-%.4f  -> debias rho med %.4f range %.4f-%.4f  -> sqrt(rho) med %.4f"%(
    np.median(v),v.min(),v.max(),np.median(rho),rho.min(),rho.max(),np.median(np.sqrt(np.clip(rho,0,1)))))
df.to_csv(r"D:\project\STImage\prototype\analysis_out\rtech_splithalf_v2.csv",index=False)
