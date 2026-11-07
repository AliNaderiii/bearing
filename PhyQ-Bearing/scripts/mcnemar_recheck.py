# Run from the PhyQ-Bearing folder: python scripts/mcnemar_recheck.py
import sys, numpy as np, os
sys.path.insert(0,'.')
from phyq.analysis.stats_tests import mcnemar, grouped_bootstrap_ci
R='results/main_tuned'
tags=['phyq_nomed','phyq_nomed_noquad','phyq_transnet','cnn_transformer','drsn','qcnn','wdcnn','stats_mlp','classbd','kurtogram_ses']
def load(ds,var,tag,s):
    p=f'{R}/{ds}/{var}/{tag}/seed{s}/test_probs.npz'
    return np.load(p) if os.path.exists(p) else None
for ds,var in [('sdust','chrono'),('ims','chrono'),('nrel','expwise')]:
    print(f'\n##### {ds}')
    for key in ['clean','white__absolute__0','white__absolute__-6','white__absolute__-10','impulsive__absolute__-6','pink__absolute__-6']:
        rows=[]
        for t in tags[1:]:
            ps=[];pd_=[]
            res=[]
            for s in range(3):
                a=load(ds,var,'phyq_nomed',s); b=load(ds,var,t,s)
                if a is None or b is None or key not in a.files or key not in b.files: continue
                y=a['y']; assert (y==b['y']).all()
                r=mcnemar(y,a[key].argmax(1),b[key].argmax(1)); res.append(r)
            if not res: continue
            # pooled across seeds
            A=np.concatenate([load(ds,var,'phyq_nomed',s)[key].argmax(1) for s in range(3)])
            B=np.concatenate([load(ds,var,t,s)[key].argmax(1) for s in range(3)])
            Y=np.concatenate([load(ds,var,'phyq_nomed',s)['y'] for s in range(3)])
            rp=mcnemar(Y,A,B)
            ps=[r.get('p', r.get('p_value')) for r in res]
            rows.append(f"  vs {t:18s} dAcc={100*((A==Y).mean()-(B==Y).mean()):+6.2f}  pooled p={rp.get('p',rp.get('p_value')):.2e}  per-seed p={['%.1e'%p for p in ps]}")
        print(' ',key); print('\n'.join(rows))
