import os
import numpy as np
import pandas as pd
from utils.utils import get_simple_loader
from datasets.dataset_generic import Generic_MIL_Dataset

def count_patches(dataset):
    dataset.load_from_h5(True)
    loader = get_simple_loader(dataset)
    patches=0
    print("slides: ",len(loader))
    patch_counts=[]
    for batch_idx, (data, label, coords, ids) in enumerate(loader):
        count=len(coords)
        patches=patches+count
        patch_counts=patch_counts+[[ids,count]]
        print("number", batch_idx, "   slide",ids,"  total patches: ",patches)
    pd.DataFrame(patch_counts,columns=["slide","patches"]).to_csv("results/patch_counts/734.csv",index=False)
    return patches


csv_path='dataset_csv/set_all_734.csv'
data_root_dir="/../mount_i/features" 
features_folder="ovarian_dataset_features_256_patches_20x"
coords_path="/mount_outputs/coords"
n_classes=2
label_dict = {'high_grade':0,'low_grade':1,'clear_cell':1,'endometrioid':1,'mucinous':1}
dataset = Generic_MIL_Dataset(csv_path = csv_path,
                        data_dir= os.path.join(data_root_dir, features_folder),
                        coords_path = coords_path,
                        shuffle = False, 
                        seed = 0, 
                        print_info = True,
                        label_dict = label_dict,
                        patient_strat=False,
                        ignore=[])
split_dataset=dataset
patches  = count_patches(split_dataset)
print("{} patches".format(patches))
