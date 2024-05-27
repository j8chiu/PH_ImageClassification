import gudhi as gd
import numpy as np
import pandas as pd
from skimage import io
import os
import pickle
import bz2
import concurrent.futures

# Paths
image_dir = 'D:\\UCSD\\2024 spring\\214\\project\\PH_ImageClassification\\data\\ISIC\\test'
label_path = 'D:\\UCSD\\2024 spring\\214\\project\\PH_ImageClassification\\data\\ISIC\\label_test.csv'
output_dir = 'D:\\UCSD\\2024 spring\\214\\project\\PH_ImageClassification\\data\\ISIC\\PDnew'

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load labels
labels = pd.read_csv(label_path)

# Define function to process each image
def process_image(row):
    image_filename = os.path.join(image_dir, f"{row['image']}.jpg")
    if os.path.exists(image_filename):
        # Load the image as grayscale
        image = io.imread(image_filename, as_gray=True)
        
        # Create cubical complex from image
        cubical_complex = gd.CubicalComplex(
            dimensions=image.shape,
            top_dimensional_cells=image.flatten()
        )
        
        # Compute the persistence diagram
        diagram = cubical_complex.persistence()
        
        # Extract labels for this image
        label_values = row.drop('image').to_dict()
        
        # Persistence data including diagram and labels
        persistence_data = (diagram, label_values)
        
        # Define a path for each diagram's pickle file in the new directory
        output_file = os.path.join(output_dir, f"{row['image']}_persistence.pkl")

        # Save with compression
        with bz2.BZ2File(output_file + '.bz2', 'wb') as f:  # Change the extension accordingly
            pickle.dump(persistence_data, f)
        
        return f"Processed and saved {row['image']}"

# Use ThreadPoolExecutor to process images in parallel
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_image, row) for index, row in labels.iterrows()]
    for future in concurrent.futures.as_completed(futures):
        print(future.result())

print("All persistence diagrams have been processed and saved individually in the specified directory.")
