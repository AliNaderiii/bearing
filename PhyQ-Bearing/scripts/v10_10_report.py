"""v10.10 report: capacity-matched (width-matched) linear twin and PU speed shift.

Reads metrics.json (and, for SDUST, test_probs.npz) under --results and prints
  1. accuracy of phyq_nomed / phyq_nomed_noquad / phyq_nomed_noquad_wide / classbd per task group,
  2. paired Wilcoxon tests PhyQ - wide twin (per group, pooled, cross-load),
  3. SDUST recording-level test (seed-averaged correctness per test recording),
  4. the PU speed-shift table (writes --out/tab_speedshift.tex).
Usage: python scripts/v10_10_report.py --results results --out docs
"""
import argparse, collections, glob, json, os
import numpy as np
from scipy.stats import wilcoxon

GROUPS = {'CWRU': ['main_tuned/cwru/within_%dHP' % i for i in range(4)], 'IMS': ['main_tuned/ims/chrono'],
          'NREL': ['main_tuned/nrel/expwise'], 'SDUST': ['main_tuned/sdust/chrono'],
          'PU5': ['main_benchmark/pu/fold%d' % i for i in range(5)], 'a2r': ['main_benchmark/pu/art2real']}
KEYS = {'clean': None, 'W0': ('white', 'absolute', '0'), 'W-6': ('white', 'absolute', '-6'),
        'W-10': ('white', 'absolute', '-10'), 'P-6': ('pink', 'absolute', '-6'), 'I-6': ('impulsive', 'absolute', '-6')}
MODELS = ['phyq_nomed', 'phyq_nomed_noquad', 'phyq_nomed_noquad_wide', 'classbd']
WIDE = 'phyq_nomed_noquad_wide'


def acc(m, k):
    if k is None:
        return m['test']['clean']['acc']
    try:
        return m['test']['noise'][k[0]][k[1]][k[2]]['acc']
    except KeyError:
        return np.nan


def runs(root, g, mod):
    out = {}
    for v in GROUPS[g]:
        for f in sorted(glob.glob(os.path.join(root, v, mod, 'seed*', 'metrics.json'))):
            out[(v, os.path.basename(os.path.dirname(f)))] = json.load(open(f))
    return out


def p_wil(d):
    d = np.asarray(d)
    return wilcoxon(d).pvalue if np.any(d != 0) else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results')
    ap.add_argument('--out', default='docs')
    a = ap.parse_args()
    R = a.results
    print('== 1. accuracy (task mean of seed means): ' + ' / '.join(KEYS))
    for g in GROUPS:
        for mod in MODELS:
            rs = runs(R, g, mod)
            if not rs:
                continue
            vals = []
            for k in KEYS.values():
                per = collections.defaultdict(list)
                for (v, s), m in rs.items():
                    per[v].append(acc(m, k))
                vals.append(np.mean([np.mean(x) for x in per.values()]))
            print('%-6s %-24s n=%2d  ' % (g, mod, len(rs)) + ' / '.join('%.1f' % x for x in vals))
    print('\n== 2. PhyQ - wide twin, paired by (task, seed)')
    pooled = collections.defaultdict(list)
    for g in GROUPS:
        A, B = runs(R, g, 'phyq_nomed'), runs(R, g, WIDE)
        ks = sorted(set(A) & set(B))
        if not ks:
            continue
        row = []
        for kn, k in KEYS.items():
            d = np.array([acc(A[x], k) - acc(B[x], k) for x in ks])
            pooled[kn] += list(d)
            row.append('%s %+.1f (%d+/%d-, p=%.3f)' % (kn, d.mean(), (d > 0).sum(), (d < 0).sum(), p_wil(d)))
        print(g, len(ks), '; '.join(row))
    for kn, d in pooled.items():
        d = np.array(d)
        print('pooled %-5s n=%d %+.2f (%d+/%d-) p=%.4f' % (kn, len(d), d.mean(), (d > 0).sum(), (d < 0).sum(), p_wil(d)))
    cl = os.path.join(R, 'cross_load_tuned', 'cwru')
    if os.path.isdir(cl):
        def load(mod):
            return {(f.split(os.sep)[-4], f.split(os.sep)[-2]): json.load(open(f))
                    for f in glob.glob(os.path.join(cl, '*', mod, 'seed*', 'metrics.json'))}
        P, L, Wd = load('phyq_nomed'), load('phyq_nomed_noquad'), load(WIDE)
        ks = sorted(set(P) & set(Wd) & set(L))
        print('\n== cross-load (%d paired runs), clean: PhyQ %.2f, Lin %.2f, wide %.2f' % (
            len(ks), *[np.mean([acc(D[x], None) for x in ks]) for D in (P, L, Wd)]))
        for nm, (X, Y) in {'PhyQ-wide': (P, Wd), 'PhyQ-Lin': (P, L), 'wide-Lin': (Wd, L)}.items():
            for kn in ['clean', 'W-6', 'I-6']:
                k = KEYS[kn]
                d = np.array([acc(X[x], k) - acc(Y[x], k) for x in ks])
                print('  %-9s %-5s %+.2f (%d+/%d-) p=%.4f' % (nm, kn, d.mean(), (d > 0).sum(), (d < 0).sum(), p_wil(d)))
    sd = os.path.join(R, 'main_tuned', 'sdust', 'chrono')
    if glob.glob(os.path.join(sd, WIDE, 'seed*', 'test_probs.npz')):
        print('\n== 3. SDUST recording-level PhyQ - wide (seed-averaged correctness per recording)')
        Z = {m: [np.load(f) for f in sorted(glob.glob(os.path.join(sd, m, 'seed*', 'test_probs.npz')))]
             for m in ['phyq_nomed', WIDE]}
        y, rec = Z['phyq_nomed'][0]['y'], Z['phyq_nomed'][0]['rec_idx']
        ur = np.unique(rec)
        rng = np.random.default_rng(0)
        for c in ['clean', 'white__absolute__0', 'white__absolute__-6', 'white__absolute__-10',
                  'pink__absolute__-6', 'impulsive__absolute__-6']:
            cor = {m: np.mean([z[c].argmax(1) == y for z in v], 0) for m, v in Z.items()}
            d = np.array([cor['phyq_nomed'][rec == r].mean() - cor[WIDE][rec == r].mean() for r in ur])
            bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
            print('  %-24s %+.1f [%+.1f, %+.1f] p=%.1e (%d recordings)' % (
                c, 100 * d.mean(), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5), p_wil(d), len(ur)))
    sp = os.path.join(R, 'pu_speed_shift', 'pu')
    if os.path.isdir(sp):
        names = {'phyq_nomed': 'PhyQ-TransNet', 'phyq_nomed_noquad': 'Linear twin', 'wdcnn': 'WDCNN', 'classbd': 'ClassBD'}
        sk = {'clean': None, 'W-6': KEYS['W-6'], 'W-10': KEYS['W-10'], 'I-6': KEYS['I-6']}
        lines = []
        print('\n== 4. PU speed shift: control (N15) / N09 (mean diff; Wilcoxon p)')
        for mod, nm in names.items():
            cells = [nm]
            for kn, k in sk.items():
                T, C = {}, {}
                for f in glob.glob(os.path.join(sp, 'sp_N15to*_f*', mod, 'seed*', 'metrics.json')):
                    parts = f.split(os.sep)
                    tgt, fold = parts[-4].split('to')[1].split('_')
                    (T if tgt == 'N09' else C)[(fold, parts[-2])] = acc(json.load(open(f)), k)
                ks = sorted(set(T) & set(C))
                d = np.array([T[x] - C[x] for x in ks])
                pv = p_wil(d)
                cell = '%.1f / %.1f (%+.1f; %s)' % (np.mean([C[x] for x in ks]), np.mean([T[x] for x in ks]), d.mean(), ('%.2f' % pv) if pv >= 0.01 else ('%.3f' % pv))
                cells.append(cell.replace('(+', '($+$').replace('(-', '($-$'))
            dg = {'N15': 0, 'N09': 0}
            for f in glob.glob(os.path.join(sp, 'sp_N15to*_f*', mod, 'seed*', 'metrics.json')):
                cm = np.array(json.load(open(f))['test']['clean']['confusion'])
                if (cm.sum(0) / cm.sum()).max() >= 0.95:
                    dg[f.split(os.sep)[-4].split('to')[1][:3]] += 1
            cells.append('%d / %d' % (dg['N15'], dg['N09']))
            print('  ' + ' | '.join(cells))
            lines.append(' & '.join(cells) + r' \\')
        tex = (r"""% generated by scripts/v10_10_report.py from results/pu_speed_shift
\begin{table}[!htbp]
\caption{Paderborn speed shift: models trained and validated at 1500~rpm (N15\_M07\_F10) on the training bearings of each bearing-disjoint fold and tested on the test bearings at 1500~rpm (control) and at 900~rpm (N09\_M07\_F10). Accuracy (\%, mean over five folds $\times$ two seeds) as control / 900~rpm, with the mean paired difference and the Wilcoxon $p$-value over the ten (fold, seed) pairs; absolute SNR referenced to the clean power of the 1500-rpm training data. Last column: runs (of ten per test speed, control / 900~rpm) that predict one class for at least 95\% of the clean test windows.}\label{tab:speedshift}
\centering\scriptsize\setlength{\tabcolsep}{3pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccccc}
\toprule
Model & Clean & White, $-6$~dB & White, $-10$~dB & Impulsive, $-6$~dB & Single-class runs \\
\midrule
""" + '\n'.join(lines) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")
        os.makedirs(a.out, exist_ok=True)
        open(os.path.join(a.out, 'tab_speedshift.tex'), 'w').write(tex)
        print('wrote', os.path.join(a.out, 'tab_speedshift.tex'))


if __name__ == '__main__':
    main()
