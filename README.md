# SML: representation geometry in mutual learning

Code and experimental results for **Controlling Representation Distances in Mutual Learning: Geometry, Alignment, and Prediction**, by Soheila Ashkezari-Toussi and Vahid-Reza Sabzevari.

This repository contains research code and data, not a manuscript or preprint.

## Contents

| Path | Contents |
| --- | --- |
| `evidence/` | Exact inputs used to regenerate the paper's tables and figures, including the ResNet-18 evaluation report |
| `code/source_snapshot/` | Training sources supplied from JupyterLab |
| `code/regenerate.py` | Table, figure, and descriptive-statistic regeneration without training |
| `code/audit_resnet18_checkpoints.py` | Metric self-tests and optional evaluation of trained models |
| `tables/`, `figures/` | Derived tables and figures |
| `evidence_manifest.json` | Checksums of the analysis evidence |
| `source_results_manifest.json` | Checksums of the retained training sources |
| `reproducibility_note.md` | Optimization protocols, aggregation conventions, and provenance limits |

## Reproduce the tables and figures

Use Python 3.10 or later in a virtual environment:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r code/requirements.txt
python code/verify_evidence.py
python code/audit_resnet18_checkpoints.py --self-test
python code/regenerate.py
```

Regeneration overwrites the derived tables, figures, and `evidence/recomputed_summary.json`. It performs no model training or dataset downloads. PDF metadata can differ between executions.

## Data selection

`evidence/` contains the records used by the analysis, rather than every exploratory result:

| Experiment | Analysis input |
| --- | --- |
| Two-peer factorial sweep | `exp2_2_2_phase_diagram.json` |
| Synthetic transition | `exp3_transition.json` |
| CIFAR-10 transition | `exp3_transition_cifar10.json` |
| Matched CIFAR-10 | `exp3_baseline_comparison_cifar10.json` |
| Matched CIFAR-100 | `exp3_cifar100_fixed.json` |
| Operating-point ablation | `exp4_operating_point_check.json` |
| Population feasibility | `exp5_k_feasibility_cifar10.json` |
| ResNet-18 training results | `exp6_resnet18.json` |
| ResNet-18 representation and numerical-validity evaluation | `audit_results.json` |

Exploratory and superseded result files are excluded. Nonfinite ResNet-18 model states are retained in the evaluation report; their accuracy values must not be interpreted as valid classification results. Use `code/regenerate.py` for the reported valid-run summaries.

The original image datasets are [CIFAR-10 and CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html). Images and training checkpoints are not redistributed here. The `.pt` files in the supplied results folder contain initialization states and are also excluded. The synthetic-data generator is in the training source snapshot.

## Training sources

The archived scripts require PyTorch and torchvision in addition to the analysis dependencies. Install versions appropriate for your CPU/CUDA environment. The original training environment is not fully pinned.

Run experiment modules from `code/source_snapshot/`, where `sml/` and `experiments/` are sibling packages. Review each script's command-line arguments and the optimization table in `reproducibility_note.md` before launching training.

Byte-identical downloaded duplicates and unused experiment revisions have been removed. Two differing downloaded variants remain (`data (1).py` and `exp3_baseline_comparison_submission_ready (1).py`), because their names do not establish chronological order. The unsuffixed files are the importable sources; the alternatives preserve unresolved provenance and are not invoked automatically. These retained sources match the supplied folder, but do not establish which historical variant produced every run. Regeneration from numerical outputs does not depend on choosing between these variants.

## Optional model evaluation

Full model evaluation requires the original trusted checkpoints and CIFAR-10 data, in this layout:

```text
PROJECT/
  sml/models_resnet18.py
  checkpoints/exp6_resnet18/*.pt
  data/
```

```bash
python code/audit_resnet18_checkpoints.py --root /absolute/path/to/PROJECT --device auto
```

The evaluator does not train models or download datasets. It writes a new report under `PROJECT/results/`. Only load trusted checkpoints: the original format uses Python object deserialization.

## Interpretation and permissions

The geometric training penalty acts on minibatch-mean squared distances; the pointwise target-error statistic is a distinct quantity. Reduced geometric minima are not guarantees of SGD convergence. Direct distances depend on feature coordinates, and the high-separation summaries are conditional on numerical validity.

Public availability does not imply an open-source license. No license has been assigned; reuse permissions remain with the respective rights holders. See `VALIDATION.md` for the checks performed on this release.
