from __future__ import print_function

import argparse
import pdb
import os
import math

# internal imports
from utils.file_utils import save_pkl, load_pkl
from utils.utils import *
from utils.core_utils import train
from utils.core_utils_sampling import train_sampling
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset

# pytorch imports
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np


def main(args):
    # create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    folds = np.arange(start, end)
    for i in folds:
        seed_torch(args.seed)
        train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, 
                csv_path='{}/splits_{}.csv'.format(args.split_dir, i))
        
        ##class_counts to be used in balanced cross entropy if enabled
        class_counts_train=dataset.count_by_class(csv_path='{}/splits_{}.csv'.format(args.split_dir, i))
        class_counts_val=dataset.count_by_class(csv_path='{}/splits_{}.csv'.format(args.split_dir, i),split='val')
        class_counts=[class_counts_train[i]+class_counts_val[i] for i in range(len(class_counts_train))]
        
        datasets = (train_dataset, val_dataset, test_dataset)
        if args.sampling:
            results, test_auc, val_auc, test_acc, val_acc  = train_sampling(datasets, i, class_counts, args)
        else:
            results, test_auc, val_auc, test_acc, val_acc  = train(datasets, i, class_counts, args)
        
        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        #write results to pkl
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, results)

    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_test_auc, 
        'val_auc': all_val_auc, 'test_acc': all_test_acc, 'val_acc' : all_val_acc})

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))

# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_root_dir', type=str, default=None, 
                    help='data directory')
parser.add_argument('--max_epochs', type=int, default=200,
                    help='maximum number of epochs to train (default: 200)')
parser.add_argument('--min_epochs', type=int, default=20,
                    help='minimum number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--label_frac', type=float, default=1.0,
                    help='fraction of training labels (default: 1.0)')
parser.add_argument('--reg', type=float, default=1e-5,
                    help='weight decay (default: 1e-5)')
parser.add_argument('--seed', type=int, default=1, 
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
parser.add_argument('--split_dir', type=str, default=None, 
                    help='manually specify the set of splits to use, ' 
                    +'instead of infering from the task and label_frac argument (default: None)')
parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
parser.add_argument('--continue_training', action='store_true', default=False, help='Continue model training from latest checkpoint')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out', type=float, default=0.25, help='dropout p=0.25')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce', 'balanced_ce'], default='ce',
                     help='slide-level classification loss function (default: ce)')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil'], default='clam_sb', 
                    help='type of model (default: clam_sb, clam w/ single attention branch)')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', help='size of model, does not affect mil')
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal',  'task_2_tumor_subtyping','custom','custom_1vsall','custom_1vsall_256','custom_1vsall_256_10x','custom_1vsall_256_20x','custom_1vsall_256_20x_histo','custom_1vsall_512_fixed','custom_nsclc_256_20x','custom_1vsall_256_20x_aug','custom_1vsall_256_20x_998','custom_1vsall_256_20x_1004_augonly','custom_1vsall_256_20x_912'])
## sampling options
parser.add_argument('--sampling', action='store_true', default=False, help='sampling for faster training')
parser.add_argument('--sampling_type', type=str, choices=['spatial','textural'],default='spatial',help='type of sampling to use')
parser.add_argument('--samples_per_epoch', type=int, default=100, help='number of patches to sample per sampling epoch')
parser.add_argument('--sampling_epochs', type=int, default=10, help='number of sampling epochs')
parser.add_argument('--sampling_random', type=float, default=0.2, help='proportion of samples which are completely random per epoch')
parser.add_argument('--sampling_neighbors', type=int, default=20, help='number of nearest neighbors to consider when resampling')
parser.add_argument('--final_sample_size',type=int,default=100,help='number of patches for final sample')
parser.add_argument('--texture_model',type=str, choices=['resnet50','levit_128s'], default='resnet50',help='model to use for feature extraction in textural sampling')
### CLAM specific options
parser.add_argument('--no_inst_cluster', action='store_true', default=False,
                     help='disable instance-level clustering')
parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', None], default=None,
                     help='instance-level clustering loss function (default: None)')
parser.add_argument('--subtyping', action='store_true', default=False, 
                     help='subtyping problem')
parser.add_argument('--bag_weight', type=float, default=0.7,
                    help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--B', type=int, default=8, help='numbr of positive/negative patches to sample for clam')
args = parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

encoding_size = 1024
settings = {'num_splits': args.k, 
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs, 
            'results_dir': args.results_dir, 
            'lr': args.lr,
            'experiment': args.exp_code,
            'reg': args.reg,
            'label_frac': args.label_frac,
            'bag_loss': args.bag_loss,
            'seed': args.seed,
            'model_type': args.model_type,
            'model_size': args.model_size,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'opt': args.opt}

if args.model_type in ['clam_sb', 'clam_mb']:
   settings.update({'bag_weight': args.bag_weight,
                    'inst_loss': args.inst_loss,
                    'B': args.B})

print('\nLoad Dataset')

if args.task == 'task_1_tumor_vs_normal':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tumor_vs_normal_dummy_clean.csv',
                            data_dir= os.path.join(args.data_root_dir, 'tumor_vs_normal_resnet_features'),
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'normal_tissue':0, 'tumor_tissue':1},
                            patient_strat=False,
                            ignore=[])
                            
elif args.task == 'custom':
    args.n_classes=5
    dataset =  Generic_MIL_Dataset(csv_path = 'dataset_csv/set_all.csv',
                            data_dir= os.path.join(args.data_root_dir, 'ovarian_dataset_features'),
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'high_grade':0,'low_grade':1,'clear_cell':2,'endometrioid':3,'mucinous':4},
                            patient_strat= False,
                            ignore=[])    
    
elif args.task == 'custom_1vsall':
    args.n_classes=2
    dataset =  Generic_MIL_Dataset(csv_path = 'dataset_csv/set_all.csv',
                            data_dir= os.path.join(args.data_root_dir, 'ovarian_dataset_features'),
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'high_grade':0,'low_grade':1,'clear_cell':1,'endometrioid':1,'mucinous':1},
                            patient_strat= False,
                            ignore=[])     
    


elif args.task == 'custom_1vsall_256':
    args.n_classes=2
    dataset =  Generic_MIL_Dataset(csv_path = 'dataset_csv/set_all.csv',
                            data_dir= os.path.join(args.data_root_dir, 'ovarian_dataset_features_256_patches'),
                            shuffle = False,
                            seed = args.seed,
                            print_info = True,
                            label_dict = {'high_grade':0,'low_grade':1,'clear_cell':1,'endometrioid':1,'mucinous':1},
                            patient_strat= False,
                            ignore=[])


elif args.task == 'custom_1vsall_256_10x':
        args.n_classes=2
        dataset =  Generic_MIL_Dataset(csv_path = 'dataset_csv/set_all.csv',
                                data_dir= os.path.join(args.data_root_dir, 'ovarian_dataset_features_256_patches_10x'),
                                shuffle = False,
                                seed = args.seed,
                                print_info = True,
                                label_dict = {'high_grade':0,'low_grade':1,'clear_cell':1,'endometrioid':1,'mucinous':1},
                                patient_strat= False,
                                ignore=[])


elif args.task == 'custom_1vsall_256_20x':
        args.n_classes=2
        dataset =  Generic_MIL_Dataset(csv_path = 'dataset_csv/set_all.csv',
                                data_dir= os.path.join(args.data_root_dir, 'ovarian_dataset_features_256_patches_20x'),
                                shuffle = False,
                                seed = args.seed,
                                print_info = True,
                                label_dict = {'high_grade':0,'low_grade':1,'clear_cell':1,'endometrioid':1,'mucinous':1},
                                patient_strat= False,
                                ignore=[])


elif args.task == 'custom_1vsall_256_20x_histo':
        args.n_classes=2
        dataset =  Generic_MIL_Dataset(csv_path = 'dataset_csv/set_all.csv',
                                data_dir= os.path.join(args.data_root_dir, 'ovarian_dataset_features_256_patches_20x_histo_pretrained'),
                                shuffle = False,
                                seed = args.seed,
                                print_info = True,
                                label_dict = {'high_grade':0,'low_grade':1,'clear_cell':1,'endometrioid':1,'mucinous':1},
                                patient_strat= False,
                                ignore=[])


elif args.task == 'custom_nsclc_256_20x':
        args.n_classes=2
        dataset =  Generic_MIL_Dataset(csv_path = 'dataset_csv/set_nsclc.csv',
                                data_dir= os.path.join(args.data_root_dir, 'nsclc'),
                                shuffle = False,
                                seed = args.seed,
                                print_info = True,
                                label_dict = {'luad':0,'lusc':1},
                                patient_strat= False,
                                ignore=[])


elif args.task == 'custom_1vsall_512_fixed':
        args.n_classes=2
        dataset =  Generic_MIL_Dataset(csv_path = 'dataset_csv/set_all.csv',
                                data_dir= os.path.join(args.data_root_dir, 'ovarian_dataset_features_512_patches'),
                                shuffle = False,
                                seed = args.seed,
                                print_info = True,
                                label_dict = {'high_grade':0,'low_grade':1,'clear_cell':1,'endometrioid':1,'mucinous':1},
                                patient_strat= False,
                                ignore=[])


elif args.task == 'custom_1vsall_256_20x_998':
    args.n_classes=2
    dataset =  Generic_MIL_Dataset(csv_path = 'dataset_csv/set_all_998.csv',
                                data_dir= os.path.join(args.data_root_dir, 'ovarian_dataset_features_256_patches_20x'),
                                shuffle = False,
                                seed = args.seed,
                                print_info = True,
                                label_dict = {'high_grade':0,'low_grade':1,'clear_cell':1,'endometrioid':1,'mucinous':1},
                                patient_strat= False,
                                ignore=[])

elif args.task == 'custom_1vsall_256_20x_912':
    args.n_classes=2
    dataset =  Generic_MIL_Dataset(csv_path = 'dataset_csv/set_all_912.csv',
                                data_dir= os.path.join(args.data_root_dir, 'ovarian_dataset_features_256_patches_20x'),
                                shuffle = False,
                                seed = args.seed,
                                print_info = True,
                                label_dict = {'high_grade':0,'low_grade':1,'clear_cell':1,'endometrioid':1,'mucinous':1},
                                patient_strat= False,
                                ignore=[])


elif args.task == 'custom_1vsall_256_20x_1004_augonly':
    args.n_classes=2
    dataset =  Generic_MIL_Dataset(csv_path = 'dataset_csv/set_all_1004.csv',
                                data_dir= os.path.join(args.data_root_dir, 'ovarian_dataset_features_256_patches_20x/transforms'),
                                shuffle = False,
                                seed = args.seed,
                                print_info = True,
                                label_dict = {'high_grade':0,'low_grade':1,'clear_cell':1,'endometrioid':1,'mucinous':1},
                                patient_strat= False,
                                ignore=[])


elif args.task == 'task_2_tumor_subtyping':
    args.n_classes=3
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tumor_subtyping_dummy_clean.csv',
                            data_dir= os.path.join(args.data_root_dir, 'tumor_subtyping_resnet_features'),
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'subtype_1':0, 'subtype_2':1, 'subtype_3':2},
                            patient_strat= False,
                            ignore=[])



elif args.task == 'custom_1vsall_256_20x_aug':
    args.n_classes=2
    dataset =  Generic_MIL_Dataset(csv_path = 'dataset_csv/set_all_aug.csv',
                            data_dir= os.path.join(args.data_root_dir, 'ovarian_dataset_features_256_patches_20x'),
                            shuffle = False,
                            seed = args.seed,
                            print_info = True,
                            label_dict = {'high_grade':0,'low_grade':1,'clear_cell':1,'endometrioid':1,'mucinous':1},
                            patient_strat= False,
                            ignore=[])

    if args.model_type in ['clam_sb', 'clam_mb']:
        assert args.subtyping 
        
else:
    raise NotImplementedError
    
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

if args.split_dir is None:
    args.split_dir = os.path.join('splits', args.task+'_{}'.format(int(args.label_frac*100)))
else:
    args.split_dir = os.path.join('splits', args.split_dir)

print('split_dir: ', args.split_dir)
assert os.path.isdir(args.split_dir)

settings.update({'split_dir': args.split_dir})


with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))        

if __name__ == "__main__":
    results = main(args)
    print("finished!")
    print("end script")

