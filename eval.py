from __future__ import print_function

import numpy as np

import argparse
import torch
import torch.nn as nn
import pdb
import os
import pandas as pd
from utils.utils import *
from math import floor
import matplotlib.pyplot as plt
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
import h5py
from utils.eval_utils import *
import cProfile, pstats
#from torch.profiler import profile, record_function, ProfilerActivity

from datasets.dataset_h5 import Dataset_All_Bags

#from streamlit import legacy_caching as caching

# Training settings
parser = argparse.ArgumentParser(description='CLAM Evaluation Script')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--pretraining_dataset',type=str,choices=['ImageNet','Histo'],default='ImageNet')
parser.add_argument('--slide_ext', type=str, default= '.svs')
parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--eval_features',default=False, action='store_true',help='extract features during sampling')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--custom_downsample', type=int, default=1)
parser.add_argument('--target_patch_size', type=int, default=-1)
parser.add_argument('--data_root_dir', type=str, default=None,
                    help='data directory')
parser.add_argument('--results_dir', type=str, default='./results',
                    help='relative path to results folder, i.e. '+
                    'the directory containing models_exp_code relative to project root (default: ./results)')
parser.add_argument('--save_exp_code', type=str, default=None,
                    help='experiment code to save eval results')
parser.add_argument('--models_exp_code', type=str, default=None,
                    help='experiment code to load trained models (directory under results_dir containing model checkpoints')
parser.add_argument('--splits_dir', type=str, default=None,
                    help='splits directory, if using custom splits other than what matches the task (default: None)')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', 
                    help='size of model (default: small)')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil'], default='clam_sb', 
                    help='type of model (default: clam_sb)')
parser.add_argument('--drop_out', type=float, default=0.25, help='dropout p=0.25')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--fold', type=int, default=-1, help='single fold to evaluate')
parser.add_argument('--micro_average', action='store_true', default=False, 
                    help='use micro_average instead of macro_avearge for multiclass AUC')
parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'], default='test')
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal',  'task_2_tumor_subtyping','custom','custom_1vsall','custom_1vsall_256_20x','custom_1vsall_256_20x_aug','custom_1vsall_256_20x_1004'])
parser.add_argument('--profile', action='store_true', default=False, 
                    help='show profile of longest running code sections')
parser.add_argument('--profile_rows', type=int, default=10, help='number of rows to show from profiler (requires --profile to show any)')
parser.add_argument('--sampling', action='store_true', default=False, help='sampling for faster evaluation')
parser.add_argument('--sampling_type', type=str, choices=['spatial','textural'],default='spatial',help='type of sampling to use')
parser.add_argument('--samples_per_epoch', type=int, default=100, help='number of patches to sample per sampling epoch')
parser.add_argument('--sampling_epochs', type=int, default=10, help='number of sampling epochs')
parser.add_argument('--sampling_random', type=float, default=0.2, help='proportion of samples which are completely random per epoch')
parser.add_argument('--sampling_random_delta',type=float, default=0.02, help='reduction in sampling_random per epoch')
parser.add_argument('--sampling_neighbors', type=int, default=20, help='number of nearest neighbors to consider when resampling')
parser.add_argument('--sampling_neighbors_delta', type=int, default=0, help='reduction in number of nearest neighbors per epoch')
parser.add_argument('--texture_model',type=str, choices=['resnet50','levit_128s'], default='resnet50',help='model to use for feature extraction in textural sampling')
parser.add_argument('--plot_sampling',action='store_true',default=False,help='Save an image showing the samples taken at each at last epoch')
parser.add_argument('--plot_sampling_gif',action='store_true',default=False,help='Save a gif showing the evolution of the samples taken')
parser.add_argument('--use_all_samples',action='store_true',default=False,help='Use every previous sample for final epoch')
parser.add_argument('--final_sample_size',type=int,default=100,help='number of patches for final sample')
parser.add_argument('--retain_best_samples',type=int,default=100,help='number of highest-attention previous samples to retain for final sample')
parser.add_argument('--initial_grid_sample',action='store_true',default=False,help='Take the initial sample to be spaced out in a grid')
parser.add_argument('--sampling_average',action='store_true',default=False,help='Take the sampling weights as averages rather than maxima to leverage more learned information')
parser.add_argument('--label_dict',type=str,help='Convert labels to numbers')
parser.add_argument('--cpu_only',action='store_true',default=False,help='Use CPU only')
args = parser.parse_args()

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.cpu_only:
    torch.cuda.is_available = lambda : False
    device=torch.device("cpu")

args.save_dir = os.path.join('./eval_results', 'EVAL_' + str(args.save_exp_code))
args.models_dir = os.path.join(args.results_dir, str(args.models_exp_code))

os.makedirs(args.save_dir, exist_ok=True)

if args.splits_dir is None:
    args.splits_dir = args.models_dir

assert os.path.isdir(args.models_dir)
assert os.path.isdir(args.splits_dir)

settings = {'task': args.task,
            'split': args.split,
            'save_dir': args.save_dir, 
            'models_dir': args.models_dir,
            'model_type': args.model_type,
            'drop_out': args.drop_out,
            'model_size': args.model_size}

with open(args.save_dir + '/eval_experiment_{}.txt'.format(args.save_exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print(settings)
if args.task == 'task_1_tumor_vs_normal':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tumor_vs_normal_dummy_clean.csv',
                            data_dir= os.path.join(args.data_root_dir, 'tumor_vs_normal_resnet_features'),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'normal_tissue':0, 'tumor_tissue':1},
                            patient_strat=False,
                            ignore=[])

    
elif args.task == 'custom':
    args.n_classes=5
    dataset =  Generic_MIL_Dataset(csv_path = 'dataset_csv/set_all.csv',
                            data_dir= os.path.join(args.data_root_dir, 'ovarian_dataset_features'),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'high_grade':0,'low_grade':1,'clear_cell':2,'endometrioid':3,'mucinous':4},
                            patient_strat= False,
                            ignore=[])    
    
    
elif args.task == 'custom_1vsall':
    args.n_classes=2
    dataset =  Generic_MIL_Dataset(csv_path = 'dataset_csv/set_all.csv',
                            data_dir= os.path.join(args.data_root_dir, 'ovarian_dataset_features'),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'high_grade':0,'low_grade':1,'clear_cell':1,'endometrioid':1,'mucinous':1},
                            patient_strat= False,
                            ignore=[])   
    
    
elif args.task == 'custom_1vsall_256_20x':
    args.n_classes=2
    dataset =  Generic_MIL_Dataset(csv_path = 'dataset_csv/set_all.csv',
                            data_dir= os.path.join(args.data_root_dir, 'ovarian_dataset_features_256_patches_20x'),
                            shuffle = False,
                            print_info = True,
                            label_dict = {'high_grade':0,'low_grade':1,'clear_cell':1,'endometrioid':1,'mucinous':1},
                            patient_strat= False,
                            ignore=[])
    

elif args.task == 'task_2_tumor_subtyping':
    args.n_classes=3
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tumor_subtyping_dummy_clean.csv',
                            data_dir= os.path.join(args.data_root_dir, 'tumor_subtyping_resnet_features'),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'subtype_1':0, 'subtype_2':1, 'subtype_3':2},
                            patient_strat= False,
                            ignore=[])


elif args.task == 'custom_1vsall_256_20x_aug':
    args.n_classes=2
    dataset =  Generic_MIL_Dataset(csv_path = 'dataset_csv/set_all_aug.csv',
                            data_dir= os.path.join(args.data_root_dir, 'ovarian_dataset_features_256_patches_20x'),
                            shuffle = False,
                            print_info = True,
                            label_dict = {'high_grade':0,'low_grade':1,'clear_cell':1,'endometrioid':1,'mucinous':1},
                            patient_strat= False,
                            ignore=[])


elif args.task == 'custom_1vsall_256_20x_1004':
    args.n_classes=2
    dataset =  Generic_MIL_Dataset(csv_path = 'dataset_csv/set_all_1004.csv',
                            data_dir= os.path.join(args.data_root_dir, 'ovarian_dataset_features_256_patches_20x'),
                            shuffle = False,
                            print_info = True,
                            label_dict = {'high_grade':0,'low_grade':1,'clear_cell':1,'endometrioid':1,'mucinous':1},
                            patient_strat= False,
                            ignore=[])


else:
    raise NotImplementedError

if args.k_start == -1:
    start = 0
else:
    start = args.k_start
if args.k_end == -1:
    end = args.k
else:
    end = args.k_end

if args.fold == -1:
    folds = range(start, end)
else:
    folds = range(args.fold, args.fold+1)
ckpt_paths = [os.path.join(args.models_dir, 's_{}_checkpoint.pt'.format(fold)) for fold in folds]
datasets_id = {'train': 0, 'val': 1, 'test': 2, 'all': -1}

def count_patches(dataset,args,ckpt):
    loader = get_simple_loader(dataset)
    patches=0
    for batch_idx, (data, label,coords,slide_id) in enumerate(loader):
        patches=patches+len(data)
        print(len(data))
        print(data)
        assert 1==2,"testing"
    return patches


def main():
    all_results = []
    all_auc = []
    all_acc = []
    for ckpt_idx in range(len(ckpt_paths)):
        if args.eval_features:
            split_dataset=Dataset_All_Bags(args.csv_path)
        else:
            if datasets_id[args.split] < 0:
                split_dataset = dataset	
            else:	
                csv_path = '{}/splits_{}.csv'.format(args.splits_dir, folds[ckpt_idx])	
                datasets = dataset.return_splits(from_id=False, csv_path=csv_path)	
                split_dataset = datasets[datasets_id[args.split]]
            
        model, patient_results, test_error, auc, df  = eval(split_dataset, args, ckpt_paths[ckpt_idx])
        all_results.append(all_results)
        all_auc.append(auc)
        print("all auc", all_auc)
        all_acc.append(1-test_error)
        print("all acc", all_acc)
        if not args.eval_features:
            df.to_csv(os.path.join(args.save_dir, 'fold_{}.csv'.format(folds[ckpt_idx])), index=False)	

    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_auc, 'test_acc': all_acc})	
    if len(folds) != args.k:	
        save_name = 'summary_partial_{}_{}.csv'.format(folds[0], folds[-1])	
    else:	
        save_name = 'summary.csv'	
    final_df.to_csv(os.path.join(args.save_dir, save_name))
    
if __name__ == "__main__":
    ## clear cache to allow timing experiments to be fair on subsequent runs
    #if args.eval_features:
        #caching.clear_cache()
    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()
        main()
        print("max gpu mem usage:",torch.cuda.max_memory_allocated())
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats(args.profile_rows)
    else:
        main()
    
