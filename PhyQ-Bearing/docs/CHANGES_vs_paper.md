> **v8:** this file described the optional re-designed `phyqnet`. The paper now keeps the author's own model
> (PhyQ-TransNet = `phyq_paper`). The list of required text edits is in `docs/PAPER_FIXES_FA.md`.

# What must change in the manuscript (driven by the new pipeline)

1. **Numbers.** All tables must be regenerated from `scripts/make_tables.py`. Numbers obtained with
   the legacy scripts (window-shuffle splits, BA zero-fill, clean-signal features, X098 keys) are
   not comparable and must not appear anywhere.
2. **Protocols section** – copy from `docs/PROTOCOLS.md`: chronological / unit-disjoint splits,
   gap, non-overlapping test windows, number of test windows per class, units per split.
3. **SNR definition** – state both relative and absolute SNR; make absolute SNR the primary
   robustness metric (relative SNR preserves the amplitude of the clean window).
4. **Baselines** – report only re-implemented baselines trained with the identical pipeline
   (WDCNN, QCNN, DRSN-CW, CNN-Transformer, Stats-MLP). Literature numbers obtained with other
   protocols go to a clearly separated, caveated table (or are dropped). Remove the MDQCNN-on-PU
   entry (that paper did not evaluate PU) and the unverifiable references [26], [47].
5. **Model selection** – describe the validation-only search and give its ranking table
   (`results/search/ranking.md`) in the supplement. No component may be kept because it helps on test.
6. **Statistics** – mean ± std over 5 seeds (PU: 5 bearing folds × seeds), Wilcoxon + Holm,
   McNemar on paired noisy test sets, grouped bootstrap CIs.
7. **Complexity** – parameters, MACs (incl. attention, see `runner._attention_macs`) and latency on
   the stated GPU, for every model.
8. **Limitations** – CWRU/IMS classes = 1–2 physical faults each (temporal not cross-unit
   generalisation); NREL has 4 test experiments; synthetic noise ≠ real plant noise.
9. **Claims** – phrase contributions as what the evidence supports (e.g. "best mean macro-F1 under
   absolute-SNR noise on k of 5 datasets, significant on …"), not as universal superiority.
10. **Reproducibility** – release code + configs + `results/*/metrics.json`; fixed seeds; environment
    info is stored in every `metrics.json`.
