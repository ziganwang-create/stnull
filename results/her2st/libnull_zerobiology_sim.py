import os
os.environ["OMP_NUM_THREADS"]="3"; os.environ["MKL_NUM_THREADS"]="3"; os.environ["OPENBLAS_NUM_THREADS"]="3"
import json, numpy as np, pandas as pd, scipy.sparse as sp
ROOT=r"D:\project\STImage\prototype"; DATA=os.path.join(ROOT,"data"); RUNS=os.path.join(ROOT,"runs","resnet50")
meta=pd.read_parquet(os.path.join(DATA,"meta.parquet"))
sections=sorted(meta.section.unique()); patients=sorted(meta.patient.unique())
rng=np.random.default_rng(20260821)
def panel(p): return json.load(open(os.path.join(RUNS,"fold_%s"%p,"genes.json"),encoding="utf-8"))
def tf(c):
    rs=c.sum(axis=1,keepdims=True); return np.log1p(1e4*c/np.maximum(rs,1.0))
def pear(A,B):
    A=A-A.mean(0,keepdims=True); B=B-B.mean(0,keepdims=True)
    sa=np.sqrt((A*A).sum(0)); sb=np.sqrt((B*B).sum(0))
    with np.errstate(invalid="ignore",divide="ignore"): r=(A*B).sum(0)/(sa*sb)
    r[(sa==0)|(sb==0)]=np.nan; return r
# cache raw section matrices in gene-name space is heavy; instead cache per fold panel counts
raw={}
for sec in sections:
    X=sp.load_npz(os.path.join(DATA,"expr_%s.npz"%sec)).tocsr()
    sg=json.load(open(os.path.join(DATA,"genes_%s.json"%sec),encoding="utf-8"))
    raw[sec]=(X,{g:i for i,g in enumerate(sg)})
res=[]
for P in patients:
    pan=panel(P); G=len(pan)
    pcs={}; libs={}
    for sec in sections:
        X,gi=raw[sec]; cols=np.array([gi.get(g,-1) for g in pan]); pres=cols>=0
        n=X.shape[0]; pc=np.zeros((n,G))
        if pres.any(): pc[:,pres]=np.asarray(X[:,cols[pres]].todense())
        pc=np.round(pc); lib=np.round(np.asarray(X.sum(1)).ravel())
        pcs[sec]=pc; libs[sec]=lib
    tr=[s for s in sections if s[0]!=P]; te=[s for s in sections if s[0]==P]
    # global p from training patients, 738 categories
    tot=np.zeros(G+1)
    for s in tr:
        tot[:G]+=pcs[s].sum(0); tot[G]+=max(libs[s].sum()-pcs[s].sum(),0)
    p=tot/tot.sum()
    for mode in ("real","sim"):
        def get(s):
            if mode=="real": return pcs[s], libs[s]
            n=libs[s].astype(np.int64)
            d=rng.multinomial(n,p)
            return d[:,:G].astype(float), n.astype(float)
        Ytr=[]; Ltr=[]
        for s in tr:
            c,l=get(s); Ytr.append(tf(c)); Ltr.append(np.log(np.maximum(l,1)))
        Ytr=np.vstack(Ytr); Ltr=np.concatenate(Ltr)
        Xd=np.column_stack([np.ones_like(Ltr),Ltr])
        coef=np.linalg.lstsq(Xd,Ytr,rcond=None)[0]
        for s in te:
            c,l=get(s); Y=tf(c); L=np.log(np.maximum(l,1))
            pred=np.column_stack([np.ones_like(L),L])@coef
            r=pear(Y,pred); res.append((mode,s,float(np.nanmedian(r))))
df=pd.DataFrame(res,columns=["mode","section","med_r"])
pv=df.pivot(index="section",columns="mode",values="med_r")
print(pv.round(4).to_string())
print("MEDIAN over 36: real %.4f  sim(zero-biology) %.4f"%(pv["real"].median(),pv["sim"].median()))
print("sim >= real in %d/36 sections"%int((pv["sim"]>=pv["real"]).sum()))
print("corr(real,sim) across sections %.3f"%np.corrcoef(pv["real"],pv["sim"])[0,1])
sd=meta.groupby("section")["lib_size"].apply(lambda v: np.std(np.log(np.maximum(v,1))))
pv["sd_loglib"]=sd
print("corr(sd log lib, sim r) %.3f ; corr(sd log lib, real r) %.3f"%(
    np.corrcoef(pv.sd_loglib,pv["sim"])[0,1], np.corrcoef(pv.sd_loglib,pv["real"])[0,1]))
pv.to_csv(r"D:\project\STImage\prototype\analysis_out\libnull_zerobiology_sim.csv")
