"""v10.10: scripts/v10_10_report.py on a synthetic results tree (no torch needed)."""
import json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _metrics(acc, conf=None):
    noise = {nt: {'absolute': {s: {'acc': acc - 5} for s in ['0', '-6', '-10']}} for nt in ['white', 'pink', 'impulsive']}
    return {'test': {'clean': {'acc': acc, 'confusion': conf or [[5, 0, 0], [0, 5, 0], [0, 0, 5]]}, 'noise': noise}}


def _write(path, m):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(m, open(path, 'w'))


def test_report_runs_and_writes_speedshift_table(tmp_path):
    r = tmp_path / 'results'
    for mod, base in [('phyq_nomed', 90), ('phyq_nomed_noquad', 85), ('phyq_nomed_noquad_wide', 86), ('classbd', 88)]:
        for i in range(4):
            for s in range(3):
                _write(str(r / f'main_tuned/cwru/within_{i}HP/{mod}/seed{s}/metrics.json'), _metrics(base + s + i))
    for mod in ['phyq_nomed', 'phyq_nomed_noquad', 'phyq_nomed_noquad_wide']:
        for p in range(3):
            for s in range(3):
                _write(str(r / f'cross_load_tuned/cwru/cross_{p}to{p + 1}/{mod}/seed{s}/metrics.json'), _metrics(80 + p + s + len(mod) % 3))
    single = [[0, 0, 5], [0, 0, 5], [0, 0, 5]]
    for mod in ['phyq_nomed', 'phyq_nomed_noquad', 'wdcnn', 'classbd']:
        for f in range(5):
            for s in range(2):
                _write(str(r / f'pu_speed_shift/pu/sp_N15toN15_f{f}/{mod}/seed{s}/metrics.json'), _metrics(60 + f + s))
                _write(str(r / f'pu_speed_shift/pu/sp_N15toN09_f{f}/{mod}/seed{s}/metrics.json'),
                       _metrics(40 + f, single if mod == 'classbd' else None))
    out = tmp_path / 'docs'
    res = subprocess.run([sys.executable, os.path.join(ROOT, 'scripts', 'v10_10_report.py'), '--results', str(r), '--out', str(out)],
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert 'cross-load (9 paired runs)' in res.stdout
    tex = (out / 'tab_speedshift.tex').read_text()
    assert r'\label{tab:speedshift}' in tex
    assert 'ClassBD' in tex and '0 / 10' in tex


def test_nrel_csv_header_and_staging_without_pandas_header(tmp_path):
    """v10.10: NREL CSV headers are read with the csv module (pyarrow access-violation work-around)."""
    import numpy as np
    sys.path.insert(0, ROOT)
    from phyq.data.datasets import nrel
    p = tmp_path / 'H1.csv'
    p.write_text('\ufefftime,AN3,AN4\n' + ''.join(f'{i},{i * 0.5},{-i}\n' for i in range(10)), encoding='utf-8')
    assert nrel._csv_header(str(p)) == ['time', 'AN3', 'AN4']
    out = nrel._stage_csv(str(p), str(tmp_path / 'stage'))
    np.testing.assert_allclose(np.load(os.path.join(out, 'AN3.npy')), np.arange(10) * 0.5)
    np.testing.assert_allclose(np.load(os.path.join(out, 'AN4.npy')), -np.arange(10))
