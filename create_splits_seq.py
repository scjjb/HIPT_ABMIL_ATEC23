import pdb
import os
import pandas as pd
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Creating splits for whole slide classification')
parser.add_argument('--label_frac', type=float, default= 1.0,
                    help='fraction of labels (default: 1)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--unique_tests', default=False, action='store_true',help='ensure all test sets are unique')
parser.add_argument('--k', type=int, default=10,
                    help='number of splits (default: 10)')
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal', 'task_2_tumor_subtyping','custom','custom_998','custom_1vsall','custom_1vsall_1004','custom_1vsall_external','nsclc','custom_1vsall_aug','custom_1vsall_partaug','custom_1vsall_newonly','custom_1vsall_998_aug'])
parser.add_argument('--val_frac', type=float, default= 0.1,
                    help='fraction of labels for validation (default: 0.1)')
parser.add_argument('--test_frac', type=float, default= 0.1,
                    help='fraction of labels for test (default: 0.1)')

args = parser.parse_args()

if args.task == 'task_1_tumor_vs_normal':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tumor_vs_normal_dummy_clean.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'normal_tissue':0, 'tumor_tissue':1},
                            patient_strat=True,
                            ignore=[])

elif args.task == 'custom':
    args.n_classes=5
    dataset =  Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/set_all.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'high_grade':0,'low_grade':1,'clear_cell':2,'endometrioid':3,'mucinous':4},
                            patient_strat= True,
                            ignore=[])    

elif args.task == 'custom_998':
    args.n_classes=5
    dataset =  Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/set_all_998.csv',
                            shuffle = False,
                            seed = args.seed,
                            print_info = True,
                            label_dict = {'high_grade':0,'low_grade':1,'clear_cell':2,'endometrioid':3,'mucinous':4},
                            patient_strat= True,
                            ignore=[])


elif args.task == 'custom_1vsall':
    args.n_classes=2
    dataset =  Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/set_all.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'high_grade':0,'low_grade':1,'clear_cell':1,'endometrioid':1,'mucinous':1},
                            patient_strat= True,
                            ignore=[])     


elif args.task == 'custom_1vsall_1004':
    args.n_classes=2
    dataset =  Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/set_all_1004.csv',
                            shuffle = False,
                            seed = args.seed,
                            print_info = True,
                            label_dict = {'high_grade':0,'low_grade':1,'clear_cell':1,'endometrioid':1,'mucinous':1},
                            patient_strat= True,
                            ignore=[])


elif args.task == 'custom_1vsall_external':
    args.n_classes=2
    dataset =  Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/set_canada.csv',
                            shuffle = False,
                            seed = args.seed,
                            print_info = True,
                            label_dict = {'high_grade':0,'low_grade':1,'clear_cell':1,'endometrioid':1,'mucinous':1}, 
                            patient_strat= True,   
                            ignore=[])


elif args.task == 'nsclc':
    args.n_classes=2
    dataset =  Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/set_nsclc.csv',
                            shuffle = False,
                            seed = args.seed,
                            print_info = True,
                            label_dict = {'luad':0,'lusc':1},
                            patient_strat= True,
                            ignore=[])


elif args.task == 'task_2_tumor_subtyping':
    args.n_classes=3
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tumor_subtyping_dummy_clean.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'subtype_1':0, 'subtype_2':1, 'subtype_3':2},
                            patient_strat= True,
                            patient_voting='maj',
                            ignore=[])



elif args.task == 'custom_1vsall_aug':
    args.n_classes=2
    dataset =  Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/set_all_aug.csv',
                            shuffle = False,
                            seed = args.seed,
                            print_info = True,
                            label_dict = {'high_grade':0,'low_grade':1,'clear_cell':1,'endometrioid':1,'mucinous':1},
                            patient_strat= True,
                            ignore=[])


elif args.task == 'custom_1vsall_998_aug':
    args.n_classes=2
    dataset =  Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/set_all_998_aug.csv',
                            shuffle = False,
                            seed = args.seed,
                            print_info = True,
                            label_dict = {'high_grade':0,'low_grade':1,'clear_cell':1,'endometrioid':1,'mucinous':1},
                            patient_strat= True,
                            ignore=[])


elif args.task == 'custom_1vsall_partaug':
    args.n_classes=2
    dataset =  Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/set_all_partaug.csv',
                            shuffle = False,
                            seed = args.seed,
                            print_info = True,
                            label_dict = {'high_grade':0,'low_grade':1,'clear_cell':1,'endometrioid':1,'mucinous':1},
                            patient_strat= True,
                            ignore=[])


elif args.task == 'custom_1vsall_newonly':
    args.n_classes=2
    dataset =  Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/all_sets_new349.csv',
                            shuffle = False,
                            seed = args.seed,
                            print_info = True,
                            label_dict = {'high_grade':0,'low_grade':1,'clear_cell':1,'endometrioid':1,'mucinous':1},
                            patient_strat= True,
                            ignore=[])

else:
    raise NotImplementedError

num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
val_num = np.round(num_slides_cls * args.val_frac).astype(int)
test_num = np.round(num_slides_cls * args.test_frac).astype(int)

if __name__ == '__main__':
    if args.label_frac > 0:
        label_fracs = [args.label_frac]
    else:
        label_fracs = [0.1, 0.25, 0.5, 0.75, 1.0]
    
    for lf in label_fracs:
        split_dir = 'splits/'+ str(args.task) + '_{}'.format(int(lf * 100))
        os.makedirs(split_dir, exist_ok=True)
        dataset.create_splits(k = args.k, val_num = val_num, test_num = test_num, label_frac=lf)
        used_test_indices=[]
        used_val_indices=[]
        for i in range(args.k):
            dataset.set_splits()
            descriptor_df = dataset.test_split_gen(return_descriptor=True)
            splits = dataset.return_splits(from_id=True)
            used_test_indices=used_test_indices+list(splits[2].slide_data['slide_id'])
            used_val_indices=used_val_indices+list(splits[1].slide_data['slide_id'])
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}.csv'.format(i)))
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}_bool.csv'.format(i)), boolean_style=True)
            descriptor_df.to_csv(os.path.join(split_dir, 'splits_{}_descriptor.csv'.format(i)))
    
    print("total test WSIs: ",len(used_test_indices))
    print("Unique test WSIs: ",len(set(used_test_indices)))
    print("total val WSIs: ",len(used_val_indices))
    print("Unique val WSIs: ",len(set(used_val_indices)))
