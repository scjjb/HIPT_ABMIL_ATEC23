import pandas as pd
import os 
import h5py
import torch

#csv_path="dataset_csv/set_all_912.csv"
#csv_path="dataset_csv/set_all_912_aug.csv"
csv_path="dataset_csv/set_new_1168.csv"
df = pd.read_csv(csv_path)
print("{} slides to process".format(len(df['slide_id'])))
print("this should be integrated with feature extraction")

destination='../mount_outputs/coords'
dest_files = os.listdir(destination)
for i,slide_id in enumerate(df['slide_id']):
    slide_id=str(slide_id)
    if slide_id+'.pt' in dest_files:
        print('skipped {}'.format(slide_id))
        continue 
    
    print("iteration {}, processing coords from slide {}".format(i,slide_id))
    file_path = os.path.join("../mount_i/features/ovarian_dataset_features_256_patches_20x",'h5_files','{}.h5'.format(slide_id))
    #file_path = os.path.join("../mount_i/features/ovarian_dataset_features_256_patches_20x/transforms",'h5_files','{}.h5'.format(slide_id[:-4]))
    with h5py.File(file_path,'r') as hdf5_file:
        coords = hdf5_file['coords'][:]
    coords=torch.from_numpy(coords)
    torch.save(coords, os.path.join(destination, slide_id+'.pt'))

print("Finished")
print("\n")
