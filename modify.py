import numpy as np
import os

npy_dir = ''

new_parent_dir = r'D:\UCSD\2024 spring\214\project\chiupart\PH_ImageClassification\data\ISIC\test'

for filename in os.listdir(npy_dir):
    if filename.endswith('.npy'):
        file_path = os.path.join(npy_dir, filename)
        data = np.load(file_path, allow_pickle=True).item()
        
        new_path = os.path.join(new_parent_dir, os.path.basename(data['image_path']))
        data['image_path'] = new_path

        np.save(file_path, data)

print("Updated all npy files.")
