## Predicting Ovarian Cancer Treatment Response in Histopathology using Hierarchical Vision Transformers and Multiple Instance Learning 
<img src="CISTIB logo.png" align="right" width="240"/>

*HIPT-ABMIL is a transformer-based approach to classifying histopathology slides which leverages spatial information for better prognostication.* 

<img src="HIPT-AMBIL-ModelDiagram-Background-min.png" align="centre" width="900"/>

This repo was created as part of an entry to the MICCAI 2023 challenge [*automated prediction of treatment effectiveness in ovarian cancer using histopathological images* (ATEC23)](https://github.com/cwwang1979/MICCAI_ATEC23challenge). The training data was made available through [TCIA](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=83593077).

HIPT-CLAM is a multiple instance learning (MIL) approach in which features are extracted from 4096x4096 pixel regions using the pretrained hierarchical transformer model [HIPT_4K](https://github.com/mahmoodlab/HIPT) and these features are aggregated to generate a slide-level representation using the attention-based multiple instance learning (ABMIL) approach [CLAM](https://github.com/mahmoodlab/CLAM). 


## Code Runs
The following code was used in producing the results submitted as part of the ATEC23 challenge.

<details>
<summary>
Data acquisition
</summary>
Before running any code, we downloaded the training data from TCIA, and turned the single-level svs files into multi-level (pyramidal) svs files using libvips. Some level of compression was necessary here to reduce file sizes, though we found compression Q90 indistinguishable from uncompressed images. Single-slide example:
  
``` shell
vips tiffsave "I:\treatment_data\2-1613704B.svs" "I:\treatment_data\pyramid_jpeg90compress\2-1613704B.svs" --compression jpeg --Q 90 --tile --pyramid
```
</details>

<details>
<summary>
Tissue region extraction
</summary>
We segmented tissue using Otsu thresholding and extracted non-overlapping 4096x4096 tissue regions:
  
``` shell
python create_patches_fp.py --source "../mount_i/treatment_data/pyramid_jpeg90compress" --save_dir "../mount_outputs/extracted_mag20x_patch4096_fp_updated" --patch_size 4096 --step_size 4096 --seg --patch --stitch --sthresh 15 --mthresh 5 --use_otsu --closing 100
``` 
</details>

<details>
<summary>
Feature extraction
</summary>
We extracted [1,192] features from each 4096x4096 region using HIPT_4K:
  
``` shell
python extract_features_fp.py --use_transforms 'HIPT' --model_type 'HIPT_4K' --data_h5_dir "../mount_outputs/extracted_mag20x_Q90_patch4096_fp_updated" --data_slide_dir "../mount_i/treatment_data/pyramid_jpeg90compress" --csv_path "dataset_csv/set_treatment.csv" --feat_dir "../mount_outputs/features/treatment_Q90_hipt4096_features_normalised_updatedsegmentation" --batch_size 1 --slide_ext .svs 
```
</details>

<details>
<summary>
Hyperparameter tuning
</summary>
Grid tuning was performed using RayTune with hyperparameter options defined within main.py. This example is from tuning fold 0 of the 5-fold cross-validation using HIPT-ABMIL: 
  
``` shell
python main.py --tuning --hardware DGX --tuning_output_file /mnt/results/tuning_results/main_treatment_Q90_betterseg_patience30mineverloss_3reps_noaugs_DGX_moreoptions_fold0.csv --num_tuning_experiments 3 --data_slide_dir "/mnt/data/ATEC_jpeg90compress" --min_epochs 0 --early_stopping --split_dir "treatment_5fold_100" --k 1 --results_dir /mnt/results --exp_code treatment_HIPTnormalised_Q90_betterseg_patience30mineverloss_3reps_noaugs_tuning_moreoptions_fold0 --subtyping --weighted_sample --bag_loss ce --task treatment --max_epochs 200 --model_type clam_sb --no_inst_cluster --log_data --csv_path 'dataset_csv/set_treatment.csv' --data_root_dir "/mnt/data" --features_folder treatment_Q90_hipt4096_features_normalised_updatedsegmentation
```
</details>

<details>
<summary>
Model training
</summary>
The best model from the 5-fold cross-validation experiment (as judged by averaged validation set cross-entropy loss across three repeats and five folds) was trained:
  
``` shell
python main.py --hardware DGX --max_patches_per_slide 15 --data_slide_dir "/mnt/data/ATEC_jpeg90compress" --min_epochs 0 --early_stopping --drop_out 0.0 --lr 0.0005 --reg 0.0001 --model_size hipt_smaller --split_dir "treatment_5fold_100" --k 5 --results_dir /mnt/results --exp_code treatment_HIPTnormalised_Q90_betterseg_15patches_drop0lr0005reg0001_modelhiptsmaller_ABMILsb_ce_20x_5fold_noaugs_bestfromsecondbigtuning --subtyping --weighted_sample --bag_loss ce --task treatment --max_epochs 1000 --model_type clam_sb --no_inst_cluster --csv_path 'dataset_csv/set_treatment.csv' --data_root_dir "/mnt/data" --features_folder treatment_Q90_hipt4096_features_normalised_updatedsegmentation
```
</details>

<details>
<summary>
Model evaluation
</summary>
The model was evaluated on the test sets of the five-fold cross validation with 100,000 iterations of bootstrapping:
  
``` shell
python eval.py --drop_out 0.0 --model_size hipt_smaller --models_exp_code treatment_HIPTnormalised_Q90_betterseg_15patches_drop0lr0005reg0001_modelhiptsmaller_ABMILsb_ce_20x_5fold_noaugs_bestfromsecondbigtuning_s1 --save_exp_code treatment_HIPTnormalised_Q90_betterseg_15patches_drop0lr0005reg0001_modelhiptsmaller_ABMILsb_ce_20x_5fold_noaugs_bestfromsecondbigtuning_bootstrapping --task treatment --model_type clam_sb --results_dir /mnt/results --data_root_dir "/mnt/data" --k 5 --features_folder "treatment_Q90_hipt4096_features_normalised_updatedsegmentation" --csv_path 'dataset_csv/set_treatment.csv' 
python bootstrapping.py --num_classes 2 --model_names  treatment_HIPTnormalised_Q90_betterseg_15patches_drop0lr0005reg0001_modelhiptsmaller_ABMILsb_ce_20x_5fold_noaugs_bestfromsecondbigtuning_bootstrapping --bootstraps 100000 --run_repeats 1 --folds 5
```

The cross-validation results for this optimal HIPT-ABMIL model were as follows:

``` shell
 Confusion Matrix:
 [[ 76  49]
 [ 29 128]]

 average ce loss:  0.4858174402095372 (not bootstrapped)
 AUC mean:  [0.8206680412411297]  AUC std:  [0.02530094639907452]
 F1 mean:  [0.7659177381223935]  F1 std:  [0.02579712919409385]
 accuracy mean:  [0.7234604255319149]  accuracy std:  [0.02667653193254119]
 balanced accuracy mean:  [0.7117468943178861]  balanced accuracy std:  [0.026864606981070703]
```
</details>

<details>
<summary>
Model comparisons
</summary>

For the model tuning, only one example run is given per model, though many were needed (one run per fold per tuning run). 
  
HIPT-CLAM - Same patches and features as HIPT-ABMIL, then:
``` shell
python main.py --model_size hipt_small --tuning --hardware DGX --tuning_output_file /mnt/results/tuning_results/main_treatment_Q90_betterseg_patience30mineverloss_3reps_noaugs_DGX_moreoptionsCLAMsbpart1_fold0.csv --num_tuning_experiments 3 --data_slide_dir "/mnt/data/ATEC_jpeg90compress" --min_epochs 0 --early_stopping --split_dir "treatment_5fold_100" --k 1 --results_dir /mnt/results --exp_code treatment_HIPTnormalised_Q90_betterseg_patience30mineverloss_3reps_noaugs_tuning_moreoptionsCLAMsbpart1_fold0 --subtyping --weighted_sample --bag_loss ce --task treatment --max_epochs 200 --model_type clam_sb --csv_path 'dataset_csv/set_treatment.csv' --data_root_dir "/mnt/data" --features_folder treatment_Q90_hipt4096_features_normalised_updatedsegmentation
python main.py --hardware DGX --max_patches_per_slide 25 --data_slide_dir "/mnt/data/ATEC_jpeg90compress" --min_epochs 0 --early_stopping --drop_out 0.25 --lr 0.001 --reg 0.001 --model_size hipt_smaller --B 6 --split_dir "treatment_5fold_100" --k 5 --results_dir /mnt/results --exp_code treatment_HIPTnormalised_Q90_betterseg_25patches_drop25lr001reg001_modelhiptsmaller_CLAMsb_B6_ce_20x_5fold_noaugs_bestfromfirstbigtuningpart1 --subtyping --weighted_sample --bag_loss ce --task treatment --max_epochs 1000 --model_type clam_sb --csv_path 'dataset_csv/set_treatment.csv' --data_root_dir "/mnt/data" --features_folder treatment_Q90_hipt4096_features_normalised_updatedsegmentation
python eval.py --drop_out 0.25 --model_size hipt_smaller --models_exp_code treatment_HIPTnormalised_Q90_betterseg_25patches_drop25lr001reg001_modelhiptsmaller_CLAMsb_B6_ce_20x_5fold_noaugs_bestfromfirstbigtuningpart1_s1 --save_exp_code treatment_HIPTnormalised_Q90_betterseg_25patches_drop25lr001reg001_modelhiptsmaller_CLAMsb_B6_ce_20x_5fold_noaugs_bestfromfirstbigtuningpart1_bootstrapping --task treatment --model_type clam_sb --results_dir /mnt/results --data_root_dir "/mnt/data" --k 5 --features_folder "treatment_Q90_hipt4096_features_normalised_updatedsegmentation" --csv_path 'dataset_csv/set_treatment.csv'
python bootstrapping.py --num_classes 2 --model_names  treatment_HIPTnormalised_Q90_betterseg_25patches_drop25lr001reg001_modelhiptsmaller_CLAMsb_B6_ce_20x_5fold_noaugs_bestfromfirstbigtuningpart1_bootstrapping --bootstraps 100000 --run_repeats 1 --folds 5
```

ResNet-ABMIL:
``` shell
python create_patches_fp.py --source "../mount_i/treatment_data/pyramid_jpeg90compress" --save_dir "../mount_outputs/extracted_mag20x_Q90_patch256_fp_updated" --patch_size 256 --step_size 256 --seg --patch --stitch --sthresh 15 --mthresh 5 --use_otsu --closing 100
python extract_features_fp.py --data_h5_dir "/mnt/data/extracted_mag20x_Q90_patch256_fp_updated" --data_slide_dir "/mnt/data/ATEC_jpeg90compress" --csv_path "dataset_csv/set_treatment.csv" --feat_dir "/mnt/results/treatment_Q90_ResNet50_features_updatedsegmentation" --batch_size 32 --slide_ext .svs
python main.py --tuning --hardware DGX --tuning_output_file /mnt/results/tuning_results/main_treatment_Q90_ABMIL_resnet50_betterseg_patience30mineverloss_3reps_noaugs_DGX_moreoptions_fold0.csv --num_tuning_experiments 3 --data_slide_dir "/mnt/data/ATEC_jpeg90compress" --min_epochs 0 --early_stopping --split_dir "treatment_5fold_100" --k 1 --results_dir /mnt/results --exp_code treatment_ABMIL_resnet50_Q90_betterseg_patience30mineverloss_3reps_tuning_moreoptions_fold0 --subtyping --weighted_sample --bag_loss ce --task treatment --max_epochs 200 --model_type clam_sb --no_inst_cluster --csv_path 'dataset_csv/set_treatment.csv' --data_root_dir "/mnt/results" --features_folder treatment_Q90_ResNet50_features_updatedsegmentation
python main.py --hardware DGX --max_patches_per_slide 6000 --data_slide_dir "/mnt/data/ATEC_jpeg90compress" --min_epochs 0 --early_stopping --drop_out 0.35 --lr 0.001 --reg 0.0001 --model_size tinier --split_dir "treatment_5fold_100" --k 5 --results_dir /mnt/results --exp_code treatment_resnetABMIL_Q90_betterseg_6000patches_drop35lr001reg0001_modeltinier_ABMILsb_ce_20x_5fold_noaugs_bestfromsecondtuning --subtyping --weighted_sample --bag_loss ce --task treatment --max_epochs 1000 --model_type clam_sb --no_inst_cluster --csv_path 'dataset_csv/set_treatment.csv' --data_root_dir "/mnt/data" --features_folder treatment_Q90_ResNet50_features_updatedsegmentation
python eval.py --drop_out 0.5 --model_size tinier --models_exp_code treatment_resnetABMIL_Q90_betterseg_6000patches_drop35lr001reg0001_modeltinier_ABMILsb_ce_20x_5fold_noaugs_bestfromsecondtuning_s1 --save_exp_code treatment_resnetABMIL_Q90_betterseg_6000patches_drop35lr001reg0001_modeltinier_ABMILsb_ce_20x_5fold_noaugs_bestfromsecondtuning_bootstrapping --task treatment --model_type clam_sb --results_dir /mnt/results --data_root_dir "/mnt/data" --k 5 --features_folder "treatment_Q90_ResNet50_features_updatedsegmentation" --csv_path 'dataset_csv/set_treatment.csv' 
python bootstrapping.py --num_classes 2 --model_names  treatment_resnetABMIL_Q90_betterseg_6000patches_drop35lr001reg0001_modeltinier_ABMILsb_ce_20x_5fold_noaugs_bestfromsecondtuning_bootstrapping --bootstraps 100000 --run_repeats 1 --folds 5
```

HistoResNet-ABMIL - Same patches as ResNet-ABMIL, then:
``` shell
python extract_features_fp.py --data_h5_dir "/mnt/data/extracted_mag20x_Q90_patch256_fp_updated" --data_slide_dir "/mnt/data/ATEC_jpeg90compress" --csv_path "dataset_csv/set_treatment.csv" --feat_dir "/mnt/results/treatment_Q90_histotrained_ResNet18_features_updatedsegmentation/"  --pretraining_dataset "Histo" --model_type resnet18 --use_transforms "HIPT" --batch_size 32 --slide_ext .svs
python main.py --model_size tiny_resnet18 --tuning --hardware DGX --tuning_output_file /mnt/results/tuning_results/main_treatment_Q90_HistoABMIL_resnet18_betterseg_patience30mineverloss_3reps_noaugs_DGX_thirdtuning_fold0.csv --num_tuning_experiments 3 --data_slide_dir "/mnt/data/ATEC_jpeg90compress" --min_epochs 0 --early_stopping --split_dir "treatment_5fold_100" --k 1 --results_dir /mnt/results --exp_code treatment_HistoABMIL_resnet18_Q90_betterseg_patience30mineverloss_3reps_thirdtuning_moreoptions_fold0 --subtyping --weighted_sample --bag_loss ce --task treatment --max_epochs 200 --model_type clam_sb --no_inst_cluster --csv_path 'dataset_csv/set_treatment.csv' --data_root_dir "/mnt/results" --features_folder treatment_Q90_histotrained_ResNet18_features_updatedsegmentation
python main.py --hardware DGX --max_patches_per_slide 3000 --data_slide_dir "/mnt/data/ATEC_jpeg90compress" --min_epochs 0 --early_stopping --drop_out 0.1 --lr 0.005 --reg 0.001 --model_size small_resnet18 --split_dir "treatment_5fold_100" --k 5 --results_dir /mnt/results --exp_code treatment_historesnet18ABMIL_Q90_betterseg_3000patches_drop1lr005reg001_modelsmallresnet18_ABMILsb_ce_20x_5fold_noaugs_bestfromthirdtuning --subtyping --weighted_sample --bag_loss ce --task treatment --max_epochs 1000 --model_type clam_sb --no_inst_cluster --csv_path 'dataset_csv/set_treatment.csv' --data_root_dir "/mnt/results" --features_folder treatment_Q90_histotrained_ResNet18_features_updatedsegmentation
python eval.py --drop_out 0.1 --model_size small_resnet18 --models_exp_code treatment_historesnet18ABMIL_Q90_betterseg_3000patches_drop1lr005reg001_modelsmallresnet18_ABMILsb_ce_20x_5fold_noaugs_bestfromthirdtuning_s1 --save_exp_code treatment_historesnet18ABMIL_Q90_betterseg_3000patches_drop1lr005reg001_modelsmallresnet18_ABMILsb_ce_20x_5fold_noaugs_bestfromthirdtuning_bootstrapping --task treatment --model_type clam_sb --results_dir /mnt/results --data_root_dir "/mnt/results" --k 5 --features_folder "treatment_Q90_histotrained_ResNet18_features_updatedsegmentation" --csv_path 'dataset_csv/set_treatment.csv' 
python bootstrapping.py --num_classes 2 --model_names  treatment_historesnet18ABMIL_Q90_betterseg_3000patches_drop1lr005reg001_modelsmallresnet18_ABMILsb_ce_20x_5fold_noaugs_bestfromthirdtuning_bootstrapping --bootstraps 100000 --run_repeats 1 --folds 5
```

</details>
  
<details>
<summary>
Challenge test set
</summary>

First, the test set images were pre-processed into pyramid svs files through the same approach as used for the training set images (though these originated as .bmp files rather than .svs files), for example:

``` shell
vips tiffsave "I:\treatment_data\2023MICCAI_testing_set\0.BMP" "I:\treatment_data\testpyramid_jpeg90compress\0.svs" --compression jpeg --Q 90 --tile --pyramid
```

Patches were selected (one per slide due to the size of these images, requiring hugher closing and lower atfilter than training data) and features extracted:
``` shell
python create_patches_fp.py --source "../mount_i/treatment_data/testpyramid_jpeg90compress" --save_dir "../mount_outputs/extracted_mag20x_patch4096_fp_testset_updated_Q90" --patch_size 4096 --step_size 4096 --seg --patch --stitch --pad_slide --sthresh 15 --mthresh 5 --use_otsu --closing 200 --atfilter 8
python extract_features_fp.py --use_transforms 'HIPT' --model_type 'HIPT_4K' --data_h5_dir "../mount_outputs/extracted_mag20x_patch4096_fp_testset_updated_Q90" --data_slide_dir "../mount_i/treatment_data/testpyramid_jpeg90compress" --csv_path "dataset_csv/set_treatment_test.csv" --feat_dir "../mount_outputs/features/treatment_hipt4096_features_normalised_test_updated_Q90patches" --batch_size 1 --slide_ext .svs
```

The hyperparameters of the best-performing model on internal data was applied to create an ensemble of four models:
``` shell
python main.py --hardware DGX --max_patches_per_slide 15 --data_slide_dir "../mount_i/treatment_data/pyramid_jpeg90compress" --min_epochs 0 --early_stopping --drop_out 0.0 --lr 0.0005 --reg 0.0001 --model_size hipt_smaller --split_dir "treatment_submission_folds" --k 4 --results_dir results --exp_code treatment_HIPTnormalised_Q90_betterseg_15patches_drop0lr0005reg0001_modelhiptsmaller_ABMILsb_ce_20x_5fold_noaugs_4fold_7525test --subtyping --weighted_sample --bag_loss ce --task treatment --max_epochs 1000 --model_type clam_sb --no_inst_cluster --csv_path 'dataset_csv/set_treatment_plus_test.csv' --data_root_dir "../mount_outputs/features/" --features_folder treatment_Q90_hipt4096_features_normalised_updatedsegmentation
```

Finally, predictions were made on the TMA challenge test set, with the median of these predictions submitted for the challenge:
``` shell
python eval.py --drop_out 0.0 --model_size hipt_smaller --models_exp_code treatment_HIPTnormalised_Q90_betterseg_15patches_drop0lr0005reg0001_modelhiptsmaller_ABMILsb_ce_20x_5fold_noaugs_4fold_7525test_s1 --save_exp_code treatment_HIPTnormalised_Q90_betterseg_15patches_drop0lr0005reg0001_modelhiptsmaller_ABMILsb_ce_20x_5fold_noaugs_4fold_7525test_Q90patchestest_bootstrapping --task treatment --model_type clam_sb --results_dir results --data_root_dir "../mount_outputs/features/" --k 4 --features_folder "treatment_Q90_hipt4096_features_normalised_updatedsegmentation" --csv_path 'dataset_csv/set_treatment_plus_test.csv'
```
</details>


## Hyperparameter Tuning Details
The full details of the hyperparameter tuning are shared below for all models.

<details>
<summary>
Details
</summary>

Five hyperparameters were tuned for all models:
- Learning rate - Sets the rate of change of model parameters trained using the Adam optimiser
- Dropout - Sets the proportion of model weights to drop in each training iteration
- Regularisation - Sets the level of weight decay in the Adam optimiser
- Attention Layer Size - Sets the dimension of the attention layer, with the subsequent hidden layer size set to half of this in HIPT-based models and a quarter in ResNet-based models (due to the greater size of the feature space)
- Patches per Slide - Set the number of patches randomly selected from each slide per training epoch

One extra hyperparameter was tuned for the HIPT-CLAM models:
- B - Sets the number of regions which are clustered in feature space during model training

All models were tuned using multiple stages of grid searches. Each configuration was repeated three times and the performance averaged to account for random variations. 
The best performance (based on the cross-entropy loss of the validation sets in 5-fold cross-validation) from earlier tuning runs were used to select hyperparamter options in later runs. 
Due to resource constraints and the larger size of the ResNet-based models compared to the HIPT-based models, fewer configurations could be evaluated per run of ResNet models. This led to ResNet models being tuned in three runs rather than two. 

**HIPT-ABMIL** (best loss - 0.339033):
|    Hyperparameter    |        First Run       |     Second Run     | Final Selection |
|:--------------------:|:----------------------:|:------------------:|:---------------:|
|     Learning Rate    |    1e-2, 1e-3, 1e-4    |  5e-3, 1e-3, 5e-4  |       5e-4      |
|        Dropout       |     0.25, 0.5, 0.75    | 0.0, 0.2, 0.4, 0.6 |       0.0       |
|    Regularisation    | 1e-1, 1e-2, 1e-3, 1e-4 |  1e-3, 1e-4, 1e-5  |       1e-4      |
| Attention Layer Size |       64, 32, 16       |      32, 16, 8     |        16       |
|   Patches per Slide  |     25, 50, 75, 100    |   15, 25, 35, 45   |        15       |

**HIPT-CLAM** (best loss - 0.334781):
|    Hyperparameter    |        First Run       |     Second Run     | Final Selection |
|:--------------------:|:----------------------:|:------------------:|:---------------:|
|     Learning Rate    |    1e-2, 1e-3, 1e-4    |  5e-3, 1e-3, 5e-4  |       1e-3      |
|        Dropout       |     0.25, 0.5, 0.75    | 0.0, 0.2, 0.4, 0.6 |       0.25      |
|    Regularisation    | 1e-1, 1e-2, 1e-3, 1e-4 |  1e-3, 1e-4, 1e-5  |       1e-3      |
| Attention Layer Size |       64, 32, 16       |        16, 8       |        16       |
|   Patches per Slide  |     25, 50, 75, 100    |   15, 25, 35, 45   |        25       |
|           B          |         4, 6, 8        |      6, 8, 10      |        6        |

**ResNet-ABMIL** (best loss - 0.544718):
|    Hyperparameter    |     First Run    |    Second Run    |     Third Run     | Final Selection |
|:--------------------:|:----------------:|:----------------:|:-----------------:|:---------------:|
|     Learning Rate    | 1e-3, 1e-4, 1e-5 | 5e-3, 1e-3, 5e-4 |     1e-3, 5e-4    |       1e-3      |
|        Dropout       |  0.25, 0.5, 0.75 | 0.15, 0.35, 0.55 |   0.3, 0.4, 0.5   |       0.35      |
|    Regularisation    | 1e-2, 1e-3, 1e-4 | 1e-3, 1e-4, 1e-5 |     1e-4, 1e-5    |       1e-4      |
| Attention Layer Size |   512, 256, 64   |   512, 256, 64   |  256, 128, 64, 32 |        64       |
|   Patches per Slide  | 2500, 5000, 7500 | 6000, 5000, 4000 | 10000, 8000, 6000 |       6000      |

**HistoResNet-ABMIL** (best loss - 0.523930):
|    Hyperparameter    |     First Run    |    Second Run    |        Third Run       | Final Selection |
|:--------------------:|:----------------:|:----------------:|:----------------------:|:---------------:|
|     Learning Rate    | 1e-3, 1e-4, 1e-5 | 5e-3, 1e-3, 5e-4 |       1e-2, 5e-3       |       5e-3      |
|        Dropout       |  0.25, 0.5, 0.75 | 0.15, 0.35, 0.55 |   0.1, 0.3, 0.5, 0.7   |       0.1       |
|    Regularisation    | 1e-2, 1e-3, 1e-4 | 1e-3, 1e-4, 1e-5 |          1e-3          |       1e-3      |
| Attention Layer Size |    128, 64, 32   |    128, 64, 32   |        256, 128        |       256       |
|   Patches per Slide  | 2500, 5000, 7500 | 2000, 4000, 6000 | 1000, 3000, 5000, 7000 |       3000      |

</details>

## Requirements
All code was ran in Linux Docker containers. Two NVIDIA GPU hardware setups have been tested:
- Desktop PC with a single NVIDIA GTX 1660 GPU and an Intel i5-4460 quad-core CPU (Python 3.7.12, pytorch 1.8.1)
- NVIDIA DGX A100 server with 8 NVIDIA A100 GPUs and 256 AMD EPYC 7742 CPUs (Python 3.10.10, pytorch 1.13.1)
  
The requirements.txt file is based on the Python 3.7.12 environment.

## Reference
This code is an extension of our [previous repository](https://github.com/scjjb/DRAS-MIL), which itself was forked from the [CLAM repository](https://github.com/mahmoodlab/CLAM) with corresponding [paper](https://www.nature.com/articles/s41551-020-00682-w). Code is also used from the [HIPT repository](https://github.com/mahmoodlab/HIPT), including pretrained model weights. This repository and the original CLAM repository are both available for non-commercial academic purposes under the GPLv3 License.
