# Release validation

Checked on 2026-10-20.

- Retained training-source files match the final JupyterLab folder supplied by the authors byte-for-byte. Superseded experiments, identical duplicates, redundant result copies, and historical manifests have been removed.
- The principal result JSON files match the supplied final results folder. The synthetic transition and ResNet-18 evaluation report match the JupyterLab records.
- Current evidence and retained-source checksums are recorded in separate manifests.
- All archived training Python files parse successfully.
- CKA and orthogonal Procrustes metric self-tests pass.
- Regeneration runs in a separate validation copy without training. All seven table/statistics files and the summary JSON match the committed analysis outputs.
- A pattern scan found no GitHub tokens, AWS access-key IDs, or private-key headers in the prepared text files. This is a limited automated scan, not a guarantee against all sensitive content.
- The public release contains code and research data only. Manuscript sources, manuscript PDFs, submission correspondence, image datasets, and checkpoint files are excluded.

These checks validate reproduction from recorded numerical results. They do not constitute a new training replication or establish an authenticated historical training-code revision for every run. No new predictive-performance claims are made.
