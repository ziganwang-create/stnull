import os, sys
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="3"
import numpy as np, pandas as pd
from pathlib import Path
PROTO=Path(r"D:\project\STImage\prototype"); sys.path.insert(0,str(PROTO))
from dump_ridge_preds import target_log1p_cp10k
from train import load_all
from metric_drg import ortho_basis, resid, colcorr, partial_r, LABELS6
meta, feats, _a,_b, X, genes_common = load_all("resnet50"); del feats,_a,_b
g2col={g:i for i,g in enumerate(genes_common)}
section=meta["section"].values.astype(object); label=meta["label"].fillna("").astype(str).values
lib=np.log(np.maximum(meta["lib_size"].to_numpy(float),1.0))
ax=np.rint(meta["arr_x"].to_numpy()).astype(int); ay=np.rint(meta["arr_y"].to_numpy()).astype(int)
ANN=["A1","B1","C1","D1","E1","F1","G2","H1"]
rows=[]
NP=200
for tag,pd_ in [("resnet50","preds"),("phikon_p50","preds_phikon_p50"),("phikon_cat","preds_phikon_cat")]:
    for P in "ABCDEFGH":
        z=np.load(PROTO/pd_/("ridge_fold_%s.npz"%P))
        rte=z["rows"]; genes=z["genes"].tolist()
        gidx=np.array([g2col[g] for g in genes]); yh_all=z["pred_panel737"].astype(np.float64)
        Y=target_log1p_cp10k(X,gidx,meta,"panel737")[rte].astype(np.float64)
        st=section[rte]
        for s2 in [s for s in pd.unique(st) if s in ANN]:
            m=st==s2; y=Y[m]; yh=yh_all[m]
            lab=label[rte][m]; lm=np.array([bool(x) for x in lab])
            yl,yhl=y[lm],yh[lm]; nl=lm.sum()
            pi=np.zeros((nl,6)); j={c:i for i,c in enumerate(LABELS6)}
            for i,c in enumerate(lab[lm]): pi[i,j[c]]=1.0
            L=lib[rte][m][lm][:,None]
            one=np.ones((nl,1))
            rf=float(np.nanmedian(partial_r(yl,yhl,one)))
            crg=float(np.nanmedian(partial_r(yl,yhl,np.hstack([one,pi]))))
            X2=np.hstack([one,L,pi]); Q=ortho_basis(X2)
            ry,ryh=resid(yl,Q),resid(yhl,Q)
            drg_t=float(np.nanmedian(colcorr(ry,ryh)))
            # poly-3 depth
            Lp=np.hstack([L,L**2,L**3]); Lp=(Lp-Lp.mean(0))/np.maximum(Lp.std(0),1e-12)
            Q3=ortho_basis(np.hstack([one,Lp,pi]))
            drg_p3=float(np.nanmedian(colcorr(resid(yl,Q3),resid(yhl,Q3))))
            # toroidal-shift null on DRG_true
            xs=ax[rte][m][lm]; ys=ay[rte][m][lm]
            pos={(xs[i],ys[i]):i for i in range(nl)}
            X0,Y0=xs.min(),ys.min(); W=xs.max()-X0+1; H=ys.max()-Y0+1
            rng=np.random.default_rng(hash(s2)%2**31)
            iid=np.empty(NP); tor=[]; 
            for b in range(NP):
                pmv=rng.permutation(nl); iid[b]=np.nanmedian(colcorr(ry,ryh[pmv]))
            shifts=[(dx,dy) for dx in range(W) for dy in range(H) if not(dx==0 and dy==0)]
            rng.shuffle(shifts)
            for (dx,dy) in shifts[:NP]:
                src=[]; dst=[]
                for i in range(nl):
                    t=pos.get(((xs[i]-X0+dx)%W+X0,(ys[i]-Y0+dy)%H+Y0))
                    if t is not None: src.append(i); dst.append(t)
                if len(src)<50: continue
                tor.append(np.nanmedian(colcorr(ry[src],ryh[dst])))
            tor=np.array(tor)
            rows.append(dict(tag=tag,section=s2,patient=P,n_lab=int(nl),
                r_full=rf,crg=crg,drg_truelib=drg_t,drg_truelib_poly3=drg_p3,
                iid_p95=np.percentile(iid,95),iid_max=iid.max(),
                tor_n=len(tor),tor_p95=np.percentile(tor,95) if len(tor) else np.nan,
                tor_max=tor.max() if len(tor) else np.nan,
                tor_sd=tor.std() if len(tor) else np.nan, iid_sd=iid.std()))
            print(rows[-1]["tag"],s2,"rf=%.4f crg=%.4f drgT=%.4f p3=%.4f iidmax=%.4f tormax=%.4f sdratio=%.2f"%(
                rf,crg,drg_t,drg_p3,rows[-1]["iid_max"],rows[-1]["tor_max"],rows[-1]["tor_sd"]/max(rows[-1]["iid_sd"],1e-9)))
D=pd.DataFrame(rows); D.to_csv(PROTO/"analysis_out"/"drg_truelib_toroidal_v2.csv",index=False)
print()
for t in ["resnet50","phikon_p50","phikon_cat"]:
    d=D[D.tag==t]
    print("%-12s r_full %.4f crg %.4f DRG_truelib %.4f (poly3 %.4f)  tor_max med %.4f"%(
        t,d.r_full.median(),d.crg.median(),d.drg_truelib.median(),d.drg_truelib_poly3.median(),d.tor_max.median()))
    print("            DRG>tor_max in %d/8 ; DRG>iid_max in %d/8 ; sd inflation med %.2fx"%(
        (d.drg_truelib>d.tor_max).sum(),(d.drg_truelib>d.iid_max).sum(),(d.tor_sd/d.iid_sd).median()))
