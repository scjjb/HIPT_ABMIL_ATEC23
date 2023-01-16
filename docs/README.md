# DRAS-MIL <img src="CISTIB logo.png" align="right" width="240"/>
**D**iscriminative **R**egion **A**ctive **S**ampling for **M**ultiple **I**nstance **L**earning

<img src="482772_spatial.gif" width="500px" align="centre" />


## Workflows
### Baseline model training
1. Whole slide tissue detection and patching (create_patches_fp.py)
1. Feature extraction (extract_features_fp.py)
1. Creation of cross-validation folds (create_splits_seq.py)
1. Model training (main.py)
1. Slide evaluation (eval.py)
1. Model evaluation (other_metrics.py)

### DRAS-MIL evaluation experiments
Here we pre-compute all features as we will run multiple experiments, so will not save time by only computing relevant features
1. Whole slide tissue detection and patching (create_patches_fp.py)
1. Feature extraction (extract_features_fp.py)
1. Slide evaluation (eval.py with --sampling)
1. Model evaluation (other_metrics.py)

## DRAS-MIL evaluation in practice
Here features are evaluated only when needed
1. Whole slide tissue detection and patching (create_patches_fp.py)
1. Slide evaluation (eval.py with --sampling and --eval_features)
1. Model evaluation (other_metrics.py)


## Reference
This code is forked from the [CLAM repository](https://github.com/mahmoodlab/CLAM) with corresponding [paper](https://www.nature.com/articles/s41551-020-00682-w). This repository and the original CLAM repository are both available for non-commercial academic purposes under the GPLv3 License.
