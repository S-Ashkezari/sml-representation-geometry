# Reproducibility note

This repository contains the recorded run-level results, the ResNet-18 evaluation report, SHA-256 checksums, analysis scripts, and archived training sources. It does not contain the original training checkpoints or CIFAR images. Analysis reads existing outputs and performs no training.

## Optimization protocol

The archived trainer uses SGD and CosineAnnealingLR, with one scheduler step after each epoch. The fixed-cohort trainer contains a warmup_epochs configuration field but does not apply a warmup phase.

| Study | Training epochs | Cosine horizon | Peers | Seeds |
|---|---:|---:|---:|---:|
| Two-peer factorial sweep | 50 | 200 | 2 | 5 |
| Three-peer transition (synthetic and CIFAR-10) | 50 | 200 | 3 | 5 |
| Finite-population study | 50 | 200 | 2, 3, 4, 5 | 5 |
| Matched CIFAR comparisons | 50 | 50 | 3 | 5 |
| Operating-point ablation | 50 | 50 | 3 | 5 |
| ResNet-18 | 100 | 100 | 3 | 3 |

The sweep scripts leave the trainer's epochs field at its default of 200 while their outer training loop runs for 50 epochs. The matched comparison, corrected ablation, and ResNet-18 scripts pass the experiment duration to the trainer. The manuscript's schedule description follows these source records. The evaluation report also records the ResNet-18 model configurations.

## Limits of the records

The model-evaluation report records the model-definition hash and software versions used for re-evaluation. These identify the analysis environment, not an authenticated historical training revision for every run. Downloaded filenames ending in '(1)' are retained as source variants and do not establish which revision generated an experiment.

The factorial JSON retains train_samples and test_samples options even when smoke_cifar is false. The archived loader applies those limits only in smoke mode. The recorded sweep has smoke_cifar=false and synthetic=false; row-level flags may be serialized as strings.

Lightweight geometric metrics average evaluation-batch means equally. The ResNet-18 re-evaluation weights all 10,000 test examples equally. Four numerically invalid model populations remain recorded in the evidence and are excluded only from valid-run summaries.
