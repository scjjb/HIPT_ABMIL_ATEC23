# DRAS-MIL <img src="CISTIB logo.png" align="right" width="240"/>
**D**iscriminative **R**egion **A**ctive **S**ampling for **M**ultiple **I**nstance **L**earning

[ArXiv Paper](https://arxiv.org/abs/2302.08867) | [Upcoming Presentation](https://spie.org/medical-imaging/presentation/Efficient-subtyping-of-ovarian-cancer-histopathology-whole-slide-images-using/12471-38?enableBackToBrowse=true)

*DRAS-MIL is a sampling approach to improve the efficiency of evaluating histopathology slides while minimising the loss of classification accuracy.* 

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

## Example code runs
Create 3 cross-validation folds:
``` shell
python create_splits_seq.py --task custom_714 --seed 1 --label_frac 1 --val_frac 0.33 --test_frac 0.33 --k 3
```
Baseline model training with 500 random hyperparameter tuning experiments:
``` shell
python main.py --coords_path "../../../../MULTIX/DATA/coords" --tuning --no_inst_cluster --num_tuning_experiments 500 --tuning_output_file tuning_results/main_custom1vsall_714_ABMILsb_ce_finaltuning.csv --split_dir /workspace/CLAM-private/splits/custom_714_100 --k 1 --results_dir /workspace/CLAM-private/results --exp_code main_custom1vsall_714_ABMILsb_ce_finaltuning --weighted_sample --bag_loss ce --task ovarian_1vsall --min_epochs 50 --max_epochs 500 --model_type clam_sb --log_data --subtyping --data_root_dir "/MULTIX/DATA/" --csv_path 'dataset_csv/set_all_714.csv' --features_folder "ovarian_dataset_features_256_patches_20x"
```
Baseline model training with the best hyperparameters found during tuning:
``` shell
python main.py --early_stopping --use_all_samples --no_inst_cluster --reg 0.00079 --drop_out 0.02 --lr 0.0038 --split_dir /workspace/CLAM-private/splits/custom_714_100 --k 3 --results_dir /workspace/CLAM-private/results --exp_code main_nosampling_reg00079_dropout02_lr0038_1vsall_714_ABMILsb_ce_last10best_mean20stopper --weighted_sample --bag_loss ce --task ovarian_1vsall --max_epochs 500 --model_type clam_sb --log_data --subtyping --data_root_dir "/MULTIX/DATA/" --csv_path 'dataset_csv/set_all_714.csv' --features_folder "ovarian_dataset_features_256_patches_20x"
``` 

Slide evaluation processing all possible patches:
``` shell
python eval.py  --drop_out 0.02 --split val --splits_dir /workspace/CLAM-private/splits/custom_714_100 --k 3 --models_exp_code main_nosampling_reg00079_dropout02_lr0038_1vsall_714_ABMILsb_ce_last10best_mean20stopper_s1 --save_exp_code main_nosampling_reg00079_dropout02_lr0038_1vsall_714_ABMILsb_ce_last10best_mean20stopper_VALIDSET --task ovarian_1vsall --model_type clam_sb --results_dir /workspace/CLAM-private/results/ --data_root_dir "/MULTIX/DATA/" --csv_path 'dataset_csv/set_all_714.csv' --features_folder "ovarian_dataset_features_256_patches_20x"
```
Bootstrapping to estimate model performance:
``` shell
python bootstrapping.py --num_classes 2 --model_names main_nosampling_reg00079_dropout02_lr0038_1vsall_714_ABMILsb_ce_last10best_mean20stopper_VALIDSET --bootstraps 1000 --run_repeats 1 --folds 3
```

## Reference
This code is forked from the [CLAM repository](https://github.com/mahmoodlab/CLAM) with corresponding [paper](https://www.nature.com/articles/s41551-020-00682-w). This repository and the original CLAM repository are both available for non-commercial academic purposes under the GPLv3 License.
