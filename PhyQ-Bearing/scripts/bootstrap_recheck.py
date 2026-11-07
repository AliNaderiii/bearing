# Run from the PhyQ-Bearing folder: python scripts/bootstrap_recheck.py
import numpy as np
rng=np.random.default_rng(0)
def L(ds,v,t,s): return np.load(f'results/main_tuned/{ds}/{v}/{t}/seed{s}/test_probs.npz')
for ds,v,bs in [('sdust','chrono',['drsn','cnn_transformer','classbd']),('ims','chrono',['classbd','cnn_transformer','wdcnn','qcnn','drsn']),('nrel','expwise',['classbd','wdcnn'])]:
    z0=L(ds,v,'phyq_nomed',0); rec=z0['rec_idx']; y=z0['y']
    recs=np.unique(rec); idx=[np.where(rec==r)[0] for r in recs]
    for key in ['white__absolute__-6','white__absolute__-10','impulsive__absolute__-6']:
        A=np.stack([(L(ds,v,'phyq_nomed',s)[key].argmax(1)==y) for s in range(3)]).astype(float)
        for b in bs:
            B=np.stack([(L(ds,v,b,s)[key].argmax(1)==y) for s in range(3)]).astype(float)
            D=(A-B).mean(0)  # per-window mean over seeds
            # per-recording sums
            sums=np.array([D[i].sum() for i in idx]); cnt=np.array([len(i) for i in idx])
            boots=[]
            for _ in range(2000):
                k=rng.integers(0,len(recs),len(recs)); boots.append(sums[k].sum()/cnt[k].sum())
            lo,hi=np.percentile(boots,[2.5,97.5])
            print(f"{ds:6s} {key:24s} nomed-{b:16s} d={100*D.mean():+6.2f}  95%CI [{100*lo:+6.2f},{100*hi:+6.2f}]  nrec={len(recs)}")
