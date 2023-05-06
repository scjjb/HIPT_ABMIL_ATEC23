import torch

normal_path = "../mount_i/features/treatment_features_256_patches_20x/pt_files/2-1613704B.pt"
aug_path = "../mount_i/features/treatment_features_256_patches_20x/pt_files/2-1613704Baug1.pt"
aug2_path = "../mount_i/features/treatment_features_256_patches_20x/pt_files/2-1613704Baug2.pt"
normal_features = torch.load(normal_path)
aug_features = torch.load(aug_path)
aug2_features = torch.load(aug2_path)
print(normal_features)
print("AUG1 FEATURES: \n") 
print(aug_features)
print("AUG2 FEATURES: \n")
print(aug2_features)
